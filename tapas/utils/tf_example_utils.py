import collections
import dataclasses
import enum
import hashlib
import random
from typing import Iterable, List, Mapping, Optional, Text, Tuple

import tensorflow as tf
from absl import logging
from official.nlp.bert import tokenization

from tapas.protos import interaction_pb2
from tapas.utils import (constants, interpretation_utils,
                         number_annotation_utils, text_utils)

_NS = 'main'
_CLS = '[CLS]'
_EMPTY = '[EMPTY]'
_MASK = '[MASK]'
_SEP = '[SEP]'
_NAN = float('nan')
_MAX_NUM_CANDIDATES = 1000
_MAX_NUM_ROWS = 32
_WP_PER_CELL = 1.5
_MAX_INDEX_LENGTH = int(_MAX_NUM_CANDIDATES * _MAX_NUM_ROWS * _WP_PER_CELL)
_MAX_NUMERIC_VALUES = number_annotation_utils.MAX_QUESTION_NUMERIC_VALUES
_MAX_INT = 2**32 - 1


@dataclasses.dataclass(frozen=True)
class Token:
    original_text: Text
    piece: Text


@dataclasses.dataclass(frozen=True)
class TrainingInstance:
    tokens: List[Token]
    segment_ids: List[int]
    column_ids: List[int]
    row_ids: List[int]
    masked_lm_positions: List[int]
    masked_lm_labels: List[Text]
    is_random_table: bool


@dataclasses.dataclass(frozen=True)
class TokenCoordinates:
    column_index: int
    row_index: int
    token_index: int


@dataclasses.dataclass
class TokenizedTable:
    rows: List[List[List[Token]]]
    selected_tokens: List[TokenCoordinates]


@dataclasses.dataclass(frozen=True)
class MaskedLmInstance:
    index: int
    label: Text


@dataclasses.dataclass(frozen=True)
class ConversionConfig:
    """Configues conversion to TF example.

    vocab_file: Bert vocab file
    max_seq_length: Max length of a sequence in word pieces.
    max_column_id: Max column id to extract.
    max_row_id: Max row id to extract.
    """
    vocab_file: Text
    max_seq_length: int
    max_column_id: int
    max_row_id: int
    strip_column_names: bool


@dataclasses.dataclass(frozen=True)
class PretrainConversionConfig(ConversionConfig):
    """Configures options speciic to pretraining data creation.

    max_predictions_per_seq: Max predictions per sequence for mask task.
    min_question_length: Min question length.
    max_question_length: Max question length.
    always_continue_cells: If true always mask entire cells.
    strip_column_names: If true, add empty strings instead of column names.
    random_seed: Random seed.
    masked_lm_prob: Percentage of tokens to mask.
    """
    max_predictions_per_seq: int
    masked_lm_prob: float
    random_seed: int
    min_question_length: int
    max_question_length: int
    always_continue_cells: bool


@dataclasses.dataclass(frozen=True)
class TrimmedConversionConfig(ConversionConfig):
    # if > 0: Trim cells so that the length is <= this value.
    # Also disables further cell trimming should thus be used with
    # 'drop_rows_to_fit' below.
    # TODO(thomasmueller) Make this a parameter of the base config.
    # TODO(thomasmueller) Consider giving this a better name.
    cell_trim_length: int = -1


@dataclasses.dataclass(frozen=True)
class ClassifierConversionConfig(TrimmedConversionConfig):
    add_aggregation_candidates: bool = False
    expand_entity_descriptions: bool = False
    entity_descriptions_sentence_limit: int = 5
    use_document_title: bool = False
    # Re-computes answer coordinates from the answer text.
    update_answer_coordinates: bool = False
    # Drop last rows if table doesn't fit within max sequence length.
    drop_rows_to_fit: bool = False


@dataclasses.dataclass(frozen=True)
class SerializedExample:
    tokens: List[Token]
    column_ids: List[int]
    row_ids: List[int]
    segment_ids: List[int]


def _get_pieces(tokens):
    return (token.piece for token in tokens)


def fingerprint(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)


def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def create_string_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))


def _is_inner_wordpiece(token):
    return token.piece.startswith('##')


def _get_cell_token_indexes(column_ids, row_ids,
                            column_id, row_id):
    for index in range(len(column_ids)):
        if (column_ids[index] - 1 == column_id and row_ids[index] - 1 == row_id):
            yield index


def _get_all_answer_ids_from_coordinates(
    column_ids,
    row_ids,
    answers_list,
):
    """Maps lists of answer coordinates to token indexes."""
    answer_ids = [0] * len(column_ids)
    found_answers = set()
    all_answers = set()
    for answers in answers_list:
        for column_index, row_index in answers:
            all_answers.add((column_index, row_index))
            for index in _get_cell_token_indexes(column_ids, row_ids, column_index,
                                                 row_index):
                found_answers.add((column_index, row_index))
                answer_ids[index] = 1

    missing_count = len(all_answers) - len(found_answers)
    return answer_ids, missing_count


def _get_all_answer_ids(
    column_ids,
    row_ids,
    questions,
):
    """Maps lists of questions with answer coordinates to token indexes."""

    def _to_coordinates(
            question,):
        return [(coords.column_index, coords.row_index)
                for coords in question.answer.answer_coordinates]

    return _get_all_answer_ids_from_coordinates(
        column_ids,
        row_ids,
        answers_list=(_to_coordinates(question) for question in questions),
    )


def _find_tokens(text, segment):
    """Return start index of segment in text or None."""
    logging.info('text: %s %s', text, segment)
    for index in range(1 + len(text) - len(segment)):
        for seg_index, seg_token in enumerate(segment):
            if text[index + seg_index].piece != seg_token.piece:
                break
        else:
            return index
    return None


def _find_answer_coordinates_from_answer_text(
    tokenized_table,
    answer_text,
):
    """Returns all occurrences of answer_text in the table."""
    logging.info('answer text: %s', answer_text)
    for row_index, row in enumerate(tokenized_table.rows):
        if row_index == 0:
            # We don't search for answers in the header.
            continue
        for col_index, cell in enumerate(row):
            token_index = _find_tokens(cell, answer_text)
            if token_index is not None:
                yield TokenCoordinates(
                    row_index=row_index,
                    column_index=col_index,
                    token_index=token_index,
                )


def _find_answer_ids_from_answer_texts(
    column_ids,
    row_ids,
    tokenized_table,
    answer_texts,
):
    """Maps question with answer texts to the first matching token indexes."""
    answer_ids = [0] * len(column_ids)
    for answer_text in answer_texts:
        for coordinates in _find_answer_coordinates_from_answer_text(
            tokenized_table,
            answer_text,
        ):
            # Maps answer coordinates to indexes this can fail if tokens / rows have
            # been pruned.
            indexes = list(
                _get_cell_token_indexes(
                    column_ids,
                    row_ids,
                    column_id=coordinates.column_index,
                    row_id=coordinates.row_index - 1,
                ))
            indexes.sort()
            coordinate_answer_ids = []
            if indexes:
                begin_index = coordinates.token_index + indexes[0]
                end_index = begin_index + len(answer_text)
                for index in indexes:
                    if index >= begin_index and index < end_index:
                        coordinate_answer_ids.append(index)
            if len(coordinate_answer_ids) == len(answer_text):
                for index in coordinate_answer_ids:
                    answer_ids[index] = 1
                break
    return answer_ids


def _get_answer_ids(column_ids, row_ids,
                    question):
    """Maps answer coordinates to token indexes."""
    answer_ids, missing_count = _get_all_answer_ids(column_ids, row_ids,
                                                    [question])

    if missing_count:
        raise ValueError("Couldn't find all answers")
    return answer_ids


class TapasTokenizer:
    """Wraps a Bert tokenizer."""

    def __init__(self, vocab_file):
        self._basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
        self._wp_tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

    def get_vocab(self):
        return self._wp_tokenizer.vocab.keys()

    def tokenize(self, text):
        if text_utils.format_text(text) == constants.EMPTY_TEXT:
            return [Token(_EMPTY, _EMPTY)]
        tokens = []
        for token in self._basic_tokenizer.tokenize(text):
            for piece in self._wp_tokenizer.tokenize(token):
                tokens.append(Token(token, piece))
        return tokens

    def convert_tokens_to_ids(self, word_pieces):
        return self._wp_tokenizer.convert_tokens_to_ids(word_pieces)

    def question_encoding_cost(self, question_tokens):
        # Two extra spots of SEP and CLS.
        return len(question_tokens) + 2


class ToTensorflowExampleBase:
    """Base class for converting interactions to TF examples."""

    def __init__(self, config):
        self._max_seq_length = config.max_seq_length
        self._max_column_id = config.max_column_id
        self._max_row_id = config.max_row_id
        self._strip_column_names = config.strip_column_names
        self._tokenizer = TapasTokenizer(config.vocab_file)

    def _tokenize_table(
        self,
        table,
    ):
        """Runs tokenizer over columns and table cell texts."""
        tokenized_rows = []
        tokenized_row = []
        for column in table.columns:
            if self._strip_column_names:
                tokenized_row.append(self._tokenizer.tokenize(''))
            else:
                tokenized_row.append(self._tokenizer.tokenize(column.text))
        tokenized_rows.append(tokenized_row)

        for row in table.rows:
            tokenized_row = []
            for cell in row.cells:
                tokenized_row.append(self._tokenizer.tokenize(cell.text))
            tokenized_rows.append(tokenized_row)

        token_coordinates = []
        for row_index, row in enumerate(tokenized_rows):
            for column_index, cell in enumerate(row):
                for token_index, _ in enumerate(cell):
                    token_coordinates.append(
                        TokenCoordinates(
                            row_index=row_index,
                            column_index=column_index,
                            token_index=token_index,
                        ))

        return TokenizedTable(
            rows=tokenized_rows,
            selected_tokens=token_coordinates,
        )

    def _get_table_values(self, table, num_columns,
                          num_rows,
                          num_tokens):
        """Iterates over partial table and returns token, col. and row indexes."""
        for tc in table.selected_tokens:
            # First row is header row.
            if tc.row_index >= num_rows + 1:
                continue
            if tc.column_index >= num_columns:
                continue
            cell = table.rows[tc.row_index][tc.column_index]
            token = cell[tc.token_index]
            word_begin_index = tc.token_index
            # Don't add partial words. Find the starting word piece and check if it
            # fits in the token budget.
            while word_begin_index >= 0 \
                    and _is_inner_wordpiece(cell[word_begin_index]):
                word_begin_index -= 1
            if word_begin_index >= num_tokens:
                continue
            yield token, tc.column_index + 1, tc.row_index

    def _serialize_text(
        self, question_tokens
    ):
        """Serialzes texts in index arrays."""
        tokens = []
        segment_ids = []
        column_ids = []
        row_ids = []

        tokens.append(Token(_CLS, _CLS))
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        for token in question_tokens:
            tokens.append(token)
            segment_ids.append(0)
            column_ids.append(0)
            row_ids.append(0)

        return tokens, segment_ids, column_ids, row_ids

    def _serialize(
        self,
        question_tokens,
        table,
        num_columns,
        num_rows,
        num_tokens,
    ):
        """Serializes table and text."""
        tokens, segment_ids, column_ids, row_ids = self._serialize_text(
            question_tokens)

        tokens.append(Token(_SEP, _SEP))
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        for token, column_id, row_id in self._get_table_values(
                table, num_columns, num_rows, num_tokens):
            tokens.append(token)
            segment_ids.append(1)
            column_ids.append(column_id)
            row_ids.append(row_id)

        return SerializedExample(
            tokens=tokens,
            segment_ids=segment_ids,
            column_ids=column_ids,
            row_ids=row_ids,
        )

    def _tokenize(self, text):
        return self._tokenizer.tokenize(text)

    def _get_token_budget(self, question_tokens):
        return self._max_seq_length - self._tokenizer.question_encoding_cost(
            question_tokens)

    def _get_table_boundaries(self,
                              table):
        """Return maximal number of rows, columns and tokens."""
        max_num_tokens = 0
        max_num_columns = 0
        max_num_rows = 0
        for tc in table.selected_tokens:
            max_num_columns = max(max_num_columns, tc.column_index + 1)
            max_num_rows = max(max_num_rows, tc.row_index + 1)
            max_num_tokens = max(max_num_tokens, tc.token_index + 1)
        max_num_columns = min(self._max_column_id, max_num_columns)
        max_num_rows = min(self._max_row_id, max_num_rows)
        return max_num_rows, max_num_columns, max_num_tokens

    def _get_table_cost(self, table, num_columns,
                        num_rows, num_tokens):
        return sum(1 for _ in self._get_table_values(table, num_columns, num_rows,
                                                     num_tokens))

    def _get_column_values(
            self, table,
            col_index):
        table_numeric_values = {}
        for row_index, row in enumerate(table.rows):
            cell = row.cells[col_index]
            if cell.HasField('numeric_value'):
                table_numeric_values[row_index] = cell.numeric_value
        return table_numeric_values

    def _add_numeric_column_ranks(self, column_ids, row_ids,
                                  table,
                                  features):
        """Adds column ranks for all numeric columns."""

        ranks = [0] * len(column_ids)
        inv_ranks = [0] * len(column_ids)

        if table:
            for col_index in range(len(table.columns)):
                table_numeric_values = self._get_column_values(
                    table, col_index)
                if not table_numeric_values:
                    continue

                try:
                    key_fn = number_annotation_utils.get_numeric_sort_key_fn(
                        table_numeric_values.values())
                except ValueError:
                    continue

                table_numeric_values = {
                    row_index: key_fn(value)
                    for row_index, value in table_numeric_values.items()
                }

                table_numeric_values_inv = collections.defaultdict(list)
                for row_index, value in table_numeric_values.items():
                    table_numeric_values_inv[value].append(row_index)

                unique_values = sorted(table_numeric_values_inv.keys())

                for rank, value in enumerate(unique_values):
                    for row_index in table_numeric_values_inv[value]:
                        for index in _get_cell_token_indexes(column_ids, row_ids, col_index,
                                                             row_index):
                            ranks[index] = rank + 1
                            inv_ranks[index] = len(unique_values) - rank

        features['column_ranks'] = create_int_feature(ranks)
        features['inv_column_ranks'] = create_int_feature(inv_ranks)

    def _get_numeric_sort_key_fn(self, table_numeric_values, value):
        """Returns the sort key function for comparing value to table values.

        The function returned will be a suitable input for the key param of the
        sort(). See number_annotation_utils._get_numeric_sort_key_fn for details.

        Args:
          table_numeric_values: Numeric values of a column
          value: Numeric value in the question.

        Returns:
          A function key function to compare column and question values.

        """
        if not table_numeric_values:
            return None
        all_values = list(table_numeric_values.values())
        all_values.append(value)
        try:
            return number_annotation_utils.get_numeric_sort_key_fn(all_values)
        except ValueError:
            return None

    def _add_numeric_relations(self, question,
                               column_ids, row_ids,
                               table,
                               features):
        """Adds numeric relation emebeddings to 'features'.

        Args:
          question: The question, numeric values are used.
          column_ids: Maps word piece position to column id.
          row_ids: Maps word piece position to row id.
          table: The table containing the numeric cell values.
          features: Output.
        """

        numeric_relations = [0 for _ in column_ids]

        # Create a dictionary that maps a table cell to the set of all relations
        # this cell has with any value in the question.
        cell_indices_to_relations = collections.defaultdict(set)
        if question is not None and table is not None:
            for numeric_value_span in question.annotations.spans:
                for value in numeric_value_span.values:
                    for column_index in range(len(table.columns)):
                        table_numeric_values = self._get_column_values(
                            table, column_index)
                        sort_key_fn = self._get_numeric_sort_key_fn(table_numeric_values,
                                                                    value)
                        if sort_key_fn is None:
                            continue
                        for row_index, cell_value in table_numeric_values.items():
                            relation = number_annotation_utils.get_numeric_relation(
                                value, cell_value, sort_key_fn)
                            if relation is not None:
                                cell_indices_to_relations[column_index, row_index].add(
                                    relation)

        # For each cell add a special feature for all its word pieces.
        for (column_index,
             row_index), relations in cell_indices_to_relations.items():
            relation_set_index = 0
            for relation in relations:
                assert relation.value >= constants.Relation.EQ.value
                relation_set_index += 2**(relation.value -
                                          constants.Relation.EQ.value)
            for cell_token_index in _get_cell_token_indexes(column_ids, row_ids,
                                                            column_index, row_index):
                numeric_relations[cell_token_index] = relation_set_index

        features['numeric_relations'] = create_int_feature(numeric_relations)

    def _add_numeric_values(self, table,
                            token_ids_dict,
                            features):
        """Adds numeric values for computation of answer loss."""
        numeric_values = [_NAN] * self._max_seq_length
        if table:
            for col_index in range(len(table.columns)):
                for row_index in range(len(table.rows)):

                    numeric_value = table.rows[row_index].cells[col_index].numeric_value
                    if not numeric_value.HasField('float_value'):
                        continue

                    float_value = numeric_value.float_value
                    if float_value == float('inf'):
                        continue

                    for index in _get_cell_token_indexes(token_ids_dict['column_ids'],
                                                         token_ids_dict['row_ids'],
                                                         col_index, row_index):
                        numeric_values[index] = float_value
        features['numeric_values'] = create_float_feature(numeric_values)

    def _add_numeric_values_scale(self, table, token_ids_dict, features):
        """Adds a scale to each token to down weigh the value of long words."""
        numeric_values_scale = [1.0] * self._max_seq_length
        if not table:
            return numeric_values_scale
        for col_index in range(len(table.columns)):
            for row_index in range(len(table.rows)):
                indices = [
                    index for index in _get_cell_token_indexes(
                        token_ids_dict['column_ids'], token_ids_dict['row_ids'],
                        col_index, row_index)
                ]
                num_indices = len(indices)
                if num_indices > 1:
                    for index in indices:
                        numeric_values_scale[index] = float(num_indices)
        features['numeric_values_scale'] = create_float_feature(
            numeric_values_scale)

    def _pad_to_seq_length(self, inputs):
        while len(inputs) < self._max_seq_length:
            inputs.append(0)

    def _to_token_ids(self, tokens):
        return self._tokenizer.convert_tokens_to_ids(_get_pieces(tokens))

    def _to_features(
            self, tokens, token_ids_dict,
            table,
            question):
        """Produces a dict of TF features."""
        tokens = list(tokens)
        token_ids_dict = {
            key: list(values) for key, values in token_ids_dict.items()
        }

        length = len(tokens)
        for values in token_ids_dict.values():
            if len(values) != length:
                raise ValueError('Inconsistent length')

        input_ids = self._to_token_ids(tokens)
        input_mask = [1] * len(input_ids)

        self._pad_to_seq_length(input_ids)
        self._pad_to_seq_length(input_mask)
        for values in token_ids_dict.values():
            self._pad_to_seq_length(values)

        assert len(input_ids) == self._max_seq_length
        assert len(input_mask) == self._max_seq_length
        for values in token_ids_dict.values():
            assert len(values) == self._max_seq_length

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(input_ids)
        features['input_mask'] = create_int_feature(input_mask)
        for key, values in sorted(token_ids_dict.items()):
            features[key] = create_int_feature(values)

        self._add_numeric_column_ranks(token_ids_dict['column_ids'],
                                       token_ids_dict['row_ids'], table, features)

        self._add_numeric_relations(question, token_ids_dict['column_ids'],
                                    token_ids_dict['row_ids'], table, features)

        self._add_numeric_values(table, token_ids_dict, features)

        self._add_numeric_values_scale(table, token_ids_dict, features)

        if table:
            features['table_id'] = create_string_feature(
                [table.table_id.encode('utf8')])
            features['table_id_hash'] = create_int_feature(
                [fingerprint(table.table_id) % _MAX_INT])
        return features


class ToTrimmedTensorflowExample(ToTensorflowExampleBase):
    """Helper that allows squeezing a table into the max seq length."""

    def __init__(self, config):
        super(ToTrimmedTensorflowExample, self).__init__(config)
        self._cell_trim_length = config.cell_trim_length

    def _get_num_columns(self, table):
        num_columns = len(table.columns)
        if num_columns >= self._max_column_id:
            raise ValueError('Too many columns')
        return num_columns

    def _get_num_rows(self, table,
                      drop_rows_to_fit):
        num_rows = len(table.rows)
        if num_rows >= self._max_row_id:
            if drop_rows_to_fit:
                num_rows = self._max_row_id - 1
            else:
                raise ValueError('Too many rows')
        return num_rows

    def _to_trimmed_features(
        self,
        question,
        table,
        question_tokens,
        tokenized_table,
        num_columns,
        num_rows,
        drop_rows_to_fit=False,
    ):
        """Finds optiomal number of table tokens to include and serializes."""
        init_num_rows = num_rows
        while True:
            num_tokens = self._get_max_num_tokens(
                question_tokens,
                tokenized_table,
                num_rows=num_rows,
                num_columns=num_columns,
            )
            if num_tokens is not None:
                # We could fit the table.
                break
            if not drop_rows_to_fit or num_rows == 0:
                raise ValueError('Sequence too long')
            # Try to drop a row to fit the table.
            num_rows -= 1
        serialized_example = self._serialize(question_tokens, tokenized_table,
                                             num_columns, num_rows, num_tokens)

        assert len(serialized_example.tokens) <= self._max_seq_length

        feature_dict = {
            'column_ids': serialized_example.column_ids,
            'row_ids': serialized_example.row_ids,
            'segment_ids': serialized_example.segment_ids,
        }
        features = self._to_features(
            serialized_example.tokens, feature_dict, table=table, question=question)
        return serialized_example, features

    def _get_max_num_tokens(
        self,
        question_tokens,
        tokenized_table,
        num_columns,
        num_rows,
    ):
        """Computes max number of tokens that can be squeezed into the budget."""
        token_budget = self._get_token_budget(question_tokens)
        _, _, max_num_tokens = self._get_table_boundaries(tokenized_table)
        if self._cell_trim_length >= 0 and max_num_tokens > self._cell_trim_length:
            max_num_tokens = self._cell_trim_length
        num_tokens = 0
        for num_tokens in range(max_num_tokens + 1):
            cost = self._get_table_cost(tokenized_table, num_columns, num_rows,
                                        num_tokens + 1)
            if cost > token_budget:
                break
        if num_tokens < max_num_tokens:
            if self._cell_trim_length >= 0:
                # We don't allow dynamic trimming if a cell_trim_length is set.
                return None
            if num_tokens == 0:
                return None
        return num_tokens


class ToClassifierTensorflowExample(ToTrimmedTensorflowExample):
    """Class for converting finetuning examples."""

    def __init__(self, config):
        super(ToClassifierTensorflowExample, self).__init__(config)
        self._add_aggregation_candidates = config.add_aggregation_candidates
        self._use_document_title = config.use_document_title
        self._update_answer_coordinates = config.update_answer_coordinates
        self._drop_rows_to_fit = config.drop_rows_to_fit

    def _tokenize_extended_question(
        self,
        question,
        table,
    ):
        """Runs tokenizer over the question text and document title if it's used."""
        question_tokens = self._tokenizer.tokenize(question.text)
        text_tokens = list(question_tokens)
        if self._use_document_title and table.document_title:
            # TODO(thomasmueller) Consider adding a different segment id.
            document_title_tokens = self._tokenizer.tokenize(
                table.document_title)
            text_tokens.append(Token(_SEP, _SEP))
            text_tokens.extend(document_title_tokens)
        return text_tokens

    def _add_question_numeric_values(self, question,
                                     features):
        """Add numeric values in the question as a fixed length float feature.

        Args:
          question: The question, numeric values are used.
          features: Output.
        """
        question_numeric_values = [_NAN] * _MAX_NUMERIC_VALUES
        count = 0
        for numeric_value_span in question.annotations.spans:
            for value in numeric_value_span.values:
                if value.HasField('float_value'):
                    if count < _MAX_NUMERIC_VALUES:
                        question_numeric_values[count] = value.float_value
                    count += 1

        features['question_numeric_values'] = create_float_feature(
            question_numeric_values)

    def convert(self, interaction,
                index):
        """Converts question at 'index' to example."""
        table = interaction.table

        num_rows = self._get_num_rows(table, self._drop_rows_to_fit)
        num_columns = self._get_num_columns(table)

        question = interaction.questions[index]
        if not interaction.questions[index].answer.is_valid:
            raise ValueError('Invalid answer')

        text_tokens = self._tokenize_extended_question(question, table)
        tokenized_table = self._tokenize_table(table)

        serialized_example, features = self._to_trimmed_features(
            question=question,
            table=table,
            question_tokens=text_tokens,
            tokenized_table=tokenized_table,
            num_columns=num_columns,
            num_rows=num_rows,
            drop_rows_to_fit=self._drop_rows_to_fit)

        column_ids = serialized_example.column_ids
        row_ids = serialized_example.row_ids

        def get_answer_ids(question):
            if self._update_answer_coordinates:
                return _find_answer_ids_from_answer_texts(
                    column_ids,
                    row_ids,
                    tokenized_table,
                    answer_texts=[
                        self._tokenizer.tokenize(at)
                        for at in question.answer.answer_texts
                    ],
                )
            return _get_answer_ids(column_ids, row_ids, question)

        answer_ids = get_answer_ids(question)
        self._pad_to_seq_length(answer_ids)
        features['label_ids'] = create_int_feature(answer_ids)

        if index == 0:
            prev_answer_ids = [0] * len(column_ids)
        else:
            prev_answer_ids = get_answer_ids(interaction.questions[index - 1],)
        self._pad_to_seq_length(prev_answer_ids)
        features['prev_label_ids'] = create_int_feature(prev_answer_ids)
        features['question_id'] = create_string_feature(
            [question.id.encode('utf8')])
        features['question_id_ints'] = create_int_feature(
            text_utils.str_to_ints(
                question.id, length=text_utils.DEFAULT_INTS_LENGTH))
        features['aggregation_function_id'] = create_int_feature(
            [question.answer.aggregation_function])
        features['classification_class_index'] = create_int_feature(
            [question.answer.class_index])

        answer = question.answer.float_value if question.answer.HasField(
            'float_value') else _NAN
        features['answer'] = create_float_feature([answer])

        self._add_question_numeric_values(question, features)

        if self._add_aggregation_candidates:
            rng = random.Random(fingerprint(question.id))

            candidates = interpretation_utils.find_candidates(
                rng, table, question)
            num_initial_candidates = len(candidates)

            candidates = [c for c in candidates if len(c.rows) < _MAX_NUM_ROWS]
            candidates = candidates[:_MAX_NUM_CANDIDATES]

            funs = [0] * _MAX_NUM_CANDIDATES
            sizes = [0] * _MAX_NUM_CANDIDATES
            indexes = []

            num_final_candidates = 0
            for index, candidate in enumerate(candidates):
                token_indexes = []
                for row in candidate.rows:
                    token_indexes += _get_cell_token_indexes(column_ids, row_ids,
                                                             candidate.column, row)
                if len(indexes) + len(serialized_example.tokens) > _MAX_INDEX_LENGTH:
                    break
                num_final_candidates += 1
                sizes[index] = len(token_indexes)
                funs[index] = candidate.agg_function
                indexes += token_indexes

            # <int>[1]
            features['cand_num'] = create_int_feature([num_final_candidates])
            # <int>[_MAX_NUM_CANDIDATES]
            features['can_aggregation_function_ids'] = create_int_feature(funs)
            # <int>[_MAX_NUM_CANDIDATES]
            features['can_sizes'] = create_int_feature(sizes)
            # <int>[_MAX_INDEX_LENGTH]
            # Actual length is sum(sizes).
            features['can_indexes'] = create_int_feature(indexes)

        return tf.train.Example(features=tf.train.Features(feature=features))

    def get_empty_example(self):
        interaction = interaction_pb2.Interaction(questions=[
            interaction_pb2.Question(id=text_utils.get_padded_question_id())
        ])
        return self.convert(interaction, index=0)
