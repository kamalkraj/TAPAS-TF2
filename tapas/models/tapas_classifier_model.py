import dataclasses
import enum
import json
from typing import Iterable, List, Optional, Text

import tensorflow as tf
import tensorflow_probability as tfp
from absl import logging

from tapas.models import segmented_tensor, tf_utils
from tapas.models.modeling import BertConfig, BertModel

EPSILON_ZERO_DIVISION = 1e-10
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0


class AverageApproximationFunction(str, enum.Enum):
    RATIO = "ratio"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


@dataclasses.dataclass
class TapasClassifierConfig:
    """Helper class for configuration of Tapas model.

    bert_config: Config object for general bert hyper params.
    init_checkpoint: Location of the model checkpoint.
    positive_weight: Weight for positive labels.
    num_aggregation_labels: The number of aggregation classes to predict.
    num_classification_labels: The number of classes to predict.
    use_answer_as_supervision: Whether to use the answer as the only supervision
      for aggregation examples.
    temperature: Scales cell logits to control the skewness of probabilities.
    agg_temperature: Scales aggregation logits to control the skewness of
      probabilities.
    use_gumbel_for_cells: Applies Gumbel-Softmax to cell selection.
    use_gumbel_for_agg: Applies Gumbel-Softmax to aggregation selection.
    average_approximation_function: Method to calculate expected average of
      cells in the relaxed case.
    cell_select_pref: Preference for cell selection in ambiguous cases.
    max_num_rows: Maximum number of rows.
    max_num_columns: Maximum number of columns.
    average_logits_per_cell: Wheher to average logits per cell.
    select_one_column: Whether to constrain the model to only select cells from
      a single column.
    allow_empty_column_selection: Allow not to select any column.
    init_cell_selection_weights_to_zero: Whether to initialize cell selection.
      weights to 0 so that the initial probabilities are 50%..
    reset_position_index_per_cell: Restart position indexes at every cell.
    """

    bert_config: BertConfig
    init_checkpoint: Text
    positive_weight: float
    num_aggregation_labels: int
    num_classification_labels: int
    use_answer_as_supervision: bool
    temperature: float
    agg_temperature: float
    use_gumbel_for_cells: bool
    use_gumbel_for_agg: bool
    average_approximation_function: AverageApproximationFunction
    cell_select_pref: Optional[float]
    max_num_rows: int
    max_num_columns: int
    average_logits_per_cell: bool
    select_one_column: bool
    allow_empty_column_selection: bool = True
    init_cell_selection_weights_to_zero: bool = False
    reset_position_index_per_cell: bool = False

    def to_json_string(self):
        """Serializes this instance to a JSON string."""

        class EnhancedJSONEncoder(json.JSONEncoder):

            def default(self, o):
                if dataclasses.is_dataclass(o):
                    return dataclasses.asdict(o)
                if isinstance(o, BertConfig):
                    return o.to_dict()
                return super().default(o)

        return json.dumps(self, indent=2, sort_keys=True, cls=EnhancedJSONEncoder)

    def to_json_file(self, json_file):
        """Serializes this instance to a JSON file."""
        with tf.io.gfile.GFile(json_file, "w") as writer:
            writer.write(self.to_json_string() + "\n")

    @classmethod
    def from_dict(cls, json_object, for_prediction=False):
        """Constructs a config from a Python dictionary of parameters."""
        json_object = dict(json_object)
        # Overwrite json bert config with config object.
        json_object["bert_config"] = BertConfig.from_dict(
            json_object["bert_config"])
        # Delete deprecated option, if present.
        # TODO See of we can filter everything that's not an argument.
        if "restrict_attention" in json_object:
            del json_object["restrict_attention"]
        if for_prediction:
            # Disable training-only option to reduce input requirements.
            json_object["use_answer_as_supervision"] = False
        return TapasClassifierConfig(**json_object)

    @classmethod
    def from_json_file(cls, json_file, for_prediction=False):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text), for_prediction=for_prediction)


def _get_probs(dist):
    """Helper to extract probability from a distribution."""
    # In tensorflow_probabiliy 0.7 this attribute was filled on __init__ method
    if dist.probs is not None:
        return dist.probs
    # From 0.8 onwards the probs is not filled and a function should be used
    return dist.probs_parameter()


class ComputeTokenLogits(tf.keras.layers.Layer):
    '''Computes logits per token.'''

    def __init__(self, hidden_size, temperature, init_cell_selection_weights_to_zero, initializer_range, **kwargs):
        super(ComputeTokenLogits, self).__init__(**kwargs)
        '''
        hidden_dim: input hidden_dim
        temperature: float Temperature for the Bernoulli distribution.
        init_cell_selection_weights_to_zero: Whether the initial weights should be
            set to 0. This ensures that all tokens have the same prior probability.
        initializer_range: float, initializer range for stddev.

        '''
        self.temperature = temperature
        self.output_weights = self.add_weight(name="output_weights", shape=[hidden_size], dtype=tf.float32, trainable=True,
                                              initializer=tf.zeros_initializer() if init_cell_selection_weights_to_zero
                                              else tf.keras.initializers.TruncatedNormal(stddev=initializer_range))
        self.output_bias = self.add_weight(name="output_bias", shape=(), trainable=True,
                                           initializer=tf.zeros_initializer())

    def call(self, inputs):
        '''
        Args:
        inputs: <float>[batch_size, seq_length, hidden_dim] Output of the
            encoder layer.

        Returns:
        <float>[batch_size, seq_length] Logits per token.
        """
        '''
        logits = (tf.einsum("bsj,j->bs", inputs, self.output_weights) +
                  self.output_bias) / self.temperature
        return logits


class ComputeColumnLogits(tf.keras.layers.Layer):
    '''Computes logits for each column.'''

    def __init__(self, hidden_size, init_cell_selection_weights_to_zero, initializer_range, allow_empty_column_selection, **kwargs):
        super(ComputeColumnLogits, self).__init__(**kwargs)
        '''
        hidden_dim: input hidden_dim
        init_cell_selection_weights_to_zero: Whether the initial weights should be
        set to 0. This is also applied to column logits, as they are used to
        select the cells. This ensures that all columns have the same prior
        probability.
        allow_empty_column_selection: Allow to select no column.
        '''
        self.column_output_weights = self.add_weight(name="column_output_weights", shape=[hidden_size], dtype=tf.float32, trainable=True,
                                                     initializer=tf.zeros_initializer() if init_cell_selection_weights_to_zero
                                                     else tf.keras.initializers.TruncatedNormal(stddev=initializer_range))
        self.column_output_bias = self.add_weight(name="column_output_bias", shape=(), trainable=True,
                                                  initializer=tf.zeros_initializer())
        self.allow_empty_column_selection = allow_empty_column_selection

    def call(self, inputs, cell_index, cell_mask):
        '''
        Args:
        inputs: <float>[batch_size, seq_length, hidden_dim] Output of the
            encoder layer.
        cell_index: segmented_tensor.IndexMap [batch_size, seq_length] Index that
        groups tokens into cells.
        cell_mask: <float>[batch_size, max_num_rows * max_num_cols] Input mask per
        cell, 1 for cells that exists in the example and 0 for padding.
        '''
        token_logits = (
            tf.einsum("bsj,j->bs", inputs, self.column_output_weights) +
            self.column_output_bias)

        # Average the logits per cell and then per column.
        # Note that by linearity it doesn't matter if we do the averaging on the
        # embeddings or on the logits. For performance we do the projection first.
        # [batch_size, max_num_cols * max_num_rows]
        cell_logits, cell_logits_index = segmented_tensor.reduce_mean(
            token_logits, cell_index)

        column_index = cell_index.project_inner(cell_logits_index)
        # [batch_size, max_num_cols]
        column_logits, out_index = segmented_tensor.reduce_sum(
            cell_logits * cell_mask, column_index)
        cell_count, _ = segmented_tensor.reduce_sum(cell_mask, column_index)
        column_logits /= cell_count + EPSILON_ZERO_DIVISION

        # Mask columns that do not appear in the example.
        is_padding = tf.logical_and(cell_count < 0.5,
                                    tf.not_equal(out_index.indices, 0))
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * \
            tf.cast(is_padding, tf.float32)

        if not self.allow_empty_column_selection:
            column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(
                tf.equal(out_index.indices, 0), tf.float32)

        return column_logits


class CalculateAggregationLogits(tf.keras.layers.Layer):

    def __init__(self, num_aggregation_labels, hidden_size_agg, initializer_range, **kwargs):
        super(CalculateAggregationLogits, self).__init__(**kwargs)
        self.output_weights_agg = self.add_weight(name="output_weights_agg", shape=[num_aggregation_labels, hidden_size_agg], dtype=tf.float32,
                                                  initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range), trainable=True)
        self.output_bias_agg = self.add_weight(name="output_bias_agg", shape=[num_aggregation_labels],
                                               trainable=True, initializer=tf.zeros_initializer())

    def call(self, output_layer_aggregation):
        """Calculates the aggregation logits.
        Args:
        output_layer_aggregation: <float32>[batch_size, hidden_size]
        Returns:
        logits_aggregation: <float32>[batch_size, num_aggregation_labels]
        """
        logits_aggregation = tf.matmul(
            output_layer_aggregation, self.output_weights_agg, transpose_b=True)
        logits_aggregation = tf.nn.bias_add(
            logits_aggregation, self.output_bias_agg)
        return logits_aggregation


class ComputeClassificationLogits(tf.keras.layers.Layer):

    def __init__(self, num_classification_labels, hidden_size, initializer_range, **kwargs):
        super(ComputeClassificationLogits, self).__init__(**kwargs)
        self.output_weights_cls = self.add_weight(name="output_weights_cls", shape=[num_classification_labels, hidden_size], dtype=tf.float32,
                                                  initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range), trainable=True)
        self.output_bias_cls = self.add_weight(name="output_bias_cls", shape=[num_classification_labels],
                                               trainable=True, initializer=tf.zeros_initializer())

    def call(self, output_layer):
        logits_cls = tf.matmul(
            output_layer, self.output_weights_cls, transpose_b=True)
        logits_cls = tf.nn.bias_add(logits_cls, self.output_bias_cls)
        return logits_cls


def single_column_cell_selection(token_logits, column_logits, label_ids,
                                 cell_index, col_index, cell_mask):
    """Computes the loss for cell selection constrained to a single column.

    The loss is a hierarchical log-likelihood. The model first predicts a column
    and then selects cells within that column (conditioned on the column). Cells
    outside the selected column are never selected.

    Args:
      token_logits: <float>[batch_size, seq_length] Logits per token.
      column_logits: <float>[batch_size, max_num_cols] Logits per column.
      label_ids: <int32>[batch_size, seq_length] Labels per token.
      cell_index: segmented_tensor.IndexMap [batch_size, seq_length] Index that
        groups tokens into cells.
      col_index: segmented_tensor.IndexMap [batch_size, seq_length] Index that
        groups tokens into columns.
      cell_mask: <float>[batch_size, max_num_rows * max_num_cols] Input mask per
        cell, 1 for cells that exists in the example and 0 for padding.

    Returns:
      selection_loss_per_example: <float>[batch_size] Loss for each example.
      logits: <float>[batch_size, seq_length] New logits which are only allowed
        to select cells in a single column. Logits outside of the most likely
        column according to `column_logits` will be set to a very low value
        (such that the probabilities are 0).
    """
    # First find the column we should select. We use the column with maximum
    # number of selected cells.
    labels_per_column, _ = segmented_tensor.reduce_sum(
        tf.cast(label_ids, tf.float32), col_index)
    column_label = tf.argmax(labels_per_column, axis=-1, output_type=tf.int32)
    # Check if there are no selected cells in the column. In that case the model
    # should predict the special column id 0, which means "select nothing".
    no_cell_selected = tf.equal(tf.reduce_max(labels_per_column, axis=-1), 0)
    column_label = tf.where(no_cell_selected, tf.zeros_like(column_label),
                            column_label)

    column_dist = tfp.distributions.Categorical(logits=column_logits)

    # Reduce the labels and logits to per-cell from per-token.
    logits_per_cell, _ = segmented_tensor.reduce_mean(token_logits, cell_index)
    _, labels_index = segmented_tensor.reduce_max(
        tf.cast(label_ids, tf.int32), cell_index)

    # Mask for the selected column.
    column_id_for_cells = cell_index.project_inner(labels_index).indices

    # Set the probs outside the selected column (selected by the *model*)
    # to 0. This ensures backwards compatibility with models that select
    # cells from multiple columns.
    selected_column_id = tf.argmax(
        column_logits, axis=-1, output_type=tf.int32)
    selected_column_mask = tf.cast(
        tf.equal(column_id_for_cells, tf.expand_dims(selected_column_id,
                                                     axis=-1)), tf.float32)
    # Never select cells with the special column id 0.
    selected_column_mask = tf.where(
        tf.equal(column_id_for_cells, 0), tf.zeros_like(selected_column_mask),
        selected_column_mask)
    logits_per_cell += CLOSE_ENOUGH_TO_LOG_ZERO * (
        1.0 - cell_mask * selected_column_mask)
    logits = segmented_tensor.gather(logits_per_cell, cell_index)

    return logits


class TAPAS(tf.keras.Model):

    def __init__(self, bert_config: dict, tapas_classifier_config: TapasClassifierConfig, max_seq_length: int):
        super(TAPAS, self).__init__()

        input_token_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        segment_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')
        column_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='column_ids')
        row_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='row_ids')
        prev_label_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='prev_label_ids')
        column_ranks = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='column_ranks')
        inv_column_ranks = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='inv_column_ranks')
        numeric_relations = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='numeric_relations')

        self.bert_config = BertConfig.from_dict(bert_config)
        self.tapas_classifier_config = tapas_classifier_config
        bert_layer = BertModel(config=self.bert_config, float_type=tf.float32)
        pooled_output, sequence_output = bert_layer(input_token_ids, input_mask, segment_ids, column_ids,
                                                    row_ids, prev_label_ids, column_ranks, inv_column_ranks,
                                                    numeric_relations)
        self.bert = tf.keras.Model(inputs=[input_token_ids, input_mask, segment_ids, column_ids,
                                           row_ids, prev_label_ids, column_ranks, inv_column_ranks,
                                           numeric_relations], outputs=[pooled_output, sequence_output])
        self.compute_token_logits = ComputeTokenLogits(self.bert_config.hidden_size, self.tapas_classifier_config.temperature,
                                                       self.tapas_classifier_config.init_cell_selection_weights_to_zero,
                                                       self.bert_config.initializer_range)
        self.compute_column_logits = ComputeColumnLogits(self.bert_config.hidden_size, self.tapas_classifier_config.init_cell_selection_weights_to_zero,
                                                         self.bert_config.initializer_range, self.tapas_classifier_config.allow_empty_column_selection)
        self.do_model_aggregation = self.tapas_classifier_config.num_aggregation_labels > 0
        self.do_model_classification = self.tapas_classifier_config.num_classification_labels > 0

        if self.do_model_aggregation:
            self.calculate_aggregation_logits = CalculateAggregationLogits(self.tapas_classifier_config.num_aggregation_labels,
                                                                           self.bert_config.hidden_size, self.bert_config.initializer_range)
        else:
            self.calculate_aggregation_logits = None
        if self.do_model_classification:
            self.compute_classification_logits = ComputeClassificationLogits(self.tapas_classifier_config.num_classification_labels,
                                                                             self.bert_config.hidden_size, self.bert_config.initializer_range)
        else:
            self.compute_classification_logits = None

    def call(self, input_token_ids, input_mask, segment_ids, column_ids, row_ids, prev_label_ids, column_ranks,
             inv_column_ranks, numeric_relations, label_ids, **kwargs):

        # Construct indices for the table.
        row_index = segmented_tensor.IndexMap(indices=tf.minimum(tf.cast(row_ids, tf.int32), self.tapas_classifier_config.max_num_rows - 1),
                                              num_segments=self.tapas_classifier_config.max_num_rows,
                                              batch_dims=1)
        col_index = segmented_tensor.IndexMap(indices=tf.minimum(tf.cast(column_ids, tf.int32), self.tapas_classifier_config.max_num_columns - 1),
                                              num_segments=self.tapas_classifier_config.max_num_columns,
                                              batch_dims=1)
        cell_index = segmented_tensor.ProductIndexMap(row_index, col_index)

        # Masks.
        # <float32>[batch_size, seq_length]
        table_mask = tf.where(row_ids > 0, tf.ones_like(row_ids),
                              tf.zeros_like(row_ids))
        input_mask_float = tf.cast(input_mask, tf.float32)
        table_mask_float = tf.cast(table_mask, tf.float32)

        # Mask for cells that exist in the table (i.e. that are not padding).
        cell_mask, _ = segmented_tensor.reduce_mean(
            input_mask_float, cell_index)

        pooled_output, sequence_output = self.bert([input_token_ids, input_mask, segment_ids, column_ids,
                                                    row_ids, prev_label_ids, column_ranks, inv_column_ranks,
                                                    numeric_relations], **kwargs)
        # Compute logits per token. These are used to select individual cells.
        logits = self.compute_token_logits(sequence_output)
        # Compute logits per column. These are used to select a column.
        if self.tapas_classifier_config.select_one_column:
            column_logits = self.compute_column_logits(
                sequence_output, cell_index, cell_mask)

        logits_cls = None
        if self.do_model_classification:
            logits_cls = self.compute_classification_logits(pooled_output)

        if self.tapas_classifier_config.average_logits_per_cell:
            logits_per_cell, _ = segmented_tensor.reduce_mean(
                logits, cell_index)
            logits = segmented_tensor.gather(logits_per_cell, cell_index)
        dist_per_token = tfp.distributions.Bernoulli(logits=logits)

        if self.tapas_classifier_config.select_one_column:
            logits = single_column_cell_selection(
                logits, column_logits, label_ids, cell_index, col_index, cell_mask)
            dist_per_token = tfp.distributions.Bernoulli(logits=logits)

        logits_aggregation = None
        if self.do_model_aggregation:
            logits_aggregation = self.calculate_aggregation_logits(
                pooled_output)

        probs = _get_probs(dist_per_token) * input_mask_float

        return logits, probs, logits_aggregation, logits_cls
