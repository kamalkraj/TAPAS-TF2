import ast
import collections
import copy
import csv
import dataclasses
import enum
import functools
import json
import os
import random
import time
from typing import DefaultDict, Iterable, List, Optional, Text

import numpy as np
import tensorflow as tf
from absl import logging

from tapas.models.modeling import BertConfig
from tapas.models.tapas_classifier_model import (TAPAS,
                                                 AverageApproximationFunction,
                                                 TapasClassifierConfig)
from tapas.protos import interaction_pb2
from tapas.utils import (number_annotation_utils, prediction_utils, text_utils,
                         tf_example_utils)

_MAX_TABLE_ID = 512
_MAX_PREDICTIONS_PER_SEQ = 20
_CELL_CLASSIFICATION_THRESHOLD = 0.5
_MAX_SEQ_LENGTH = 512


SQA = {"num_aggregation_labels": 0,
       "do_model_aggregation": False,
       "use_answer_as_supervision": False,
       "init_cell_selection_weights_to_zero": False,
       "select_one_column": True,
       "allow_empty_column_selection": False,
       "cell_select_pref": None,
       "temperature": 1.0
       }

WTQ = {"num_aggregation_labels": 4,
       "do_model_aggregation": True,
       "use_answer_as_supervision": True,
       "init_cell_selection_weights_to_zero": True,
       "select_one_column": True,
       "allow_empty_column_selection": False,
       "cell_select_pref": 0.207951,
       "temperature": 0.0352513
       }

WIKISQL = {"num_aggregation_labels": 4,
           "do_model_aggregation": True,
           "use_answer_as_supervision": True,
           "init_cell_selection_weights_to_zero": False,
           "select_one_column": False,
           "allow_empty_column_selection": False,
           "cell_select_pref": 0.611754,
           "temperature": 0.107515
           }

TASKS = {"SQA": SQA, "WTQ": WTQ, "WIKISQL": WIKISQL}

os.makedirs("temp", exist_ok=True)


def get_config(task: Text, bert_config: DefaultDict, init_checkpoint: Text):
    task_params = TASKS[task.upper()]
    tapas_config = TapasClassifierConfig(bert_config=bert_config,
                                         init_checkpoint=init_checkpoint,
                                         positive_weight=10.0,
                                         num_aggregation_labels=task_params["num_aggregation_labels"],
                                         num_classification_labels=0,
                                         use_answer_as_supervision=task_params["use_answer_as_supervision"],
                                         temperature=task_params["temperature"],
                                         agg_temperature=1.0,
                                         use_gumbel_for_cells=False,
                                         use_gumbel_for_agg=False,
                                         average_approximation_function=AverageApproximationFunction.RATIO,
                                         cell_select_pref=task_params["cell_select_pref"],
                                         max_num_rows=64,
                                         max_num_columns=32,
                                         average_logits_per_cell=False,
                                         init_cell_selection_weights_to_zero=task_params[
                                             "init_cell_selection_weights_to_zero"],
                                         select_one_column=task_params["select_one_column"],
                                         allow_empty_column_selection=task_params["allow_empty_column_selection"],
                                         reset_position_index_per_cell=bert_config[
                                             "reset_position_index_per_cell"]
                                         )
    return tapas_config


def get_model(tapas_config: TapasClassifierConfig, max_seq_length=_MAX_SEQ_LENGTH):
    model = TAPAS(bert_config=tapas_config.bert_config,
                  tapas_classifier_config=tapas_config, max_seq_length=max_seq_length)
    model.load_weights(os.path.join(tapas_config.init_checkpoint, "model"))
    return model


def get_outputs(model: TAPAS, features: dict):
    logits, probs, logits_aggregation, logits_cls = model(features['input_ids'], features['input_mask'], features['segment_ids'],
                                                          features['column_ids'], features['row_ids'], features['prev_label_ids'],
                                                          features['column_ranks'], features['inv_column_ranks'],
                                                          features['numeric_relations'], features['label_ids'], training=False)
    predictions = {
        "probabilities": probs,
        "input_ids": features['input_ids'],
        "column_ids": features['column_ids'],
        "row_ids": features['row_ids'],
        "segment_ids": features['segment_ids'],
        "question_id_ints": features["question_id_ints"],
    }
    if "question_id" in features:
        # Only available when predicting on GPU.
        predictions["question_id"] = features["question_id"]
    if model.do_model_aggregation:
        predictions.update({
            "gold_aggr":
                features["aggregation_function_id"],
            "pred_aggr":
                tf.argmax(
                    logits_aggregation,
                    axis=-1,
                    output_type=tf.int32,
                )
        })
    if model.do_model_classification:
        predictions.update({
            "gold_cls":
                features["classification_class_index"],
            "pred_cls":
                tf.argmax(
                    logits_cls,
                    axis=-1,
                    output_type=tf.int32,
                )
        })
        if model.tapas_classifier_config.num_classification_labels == 2:
            predictions.update({
                "logits_cls": logits_cls[:, 1] - logits_cls[:, 0]
            })
        else:
            predictions.update({"logits_cls": logits_cls})
    return predictions


def compute_prediction_sequence(model: TAPAS, features: List[dict]):
    """Computes predictions using model's answers to the previous questions."""
    examples_by_position = collections.defaultdict(dict)
    for feature in features:
        question_id = feature["question_id"][0, 0].numpy().decode("utf-8")
        table_id, annotator, position = text_utils.parse_question_id(
            question_id)
        example_id = (table_id, annotator)
        examples_by_position[position][example_id] = feature

    all_results = []
    prev_answers = None

    for position in range(len(examples_by_position)):
        results = []
        examples = copy.deepcopy(examples_by_position[position])
        if prev_answers is not None:
            for example_id in examples:
                coords_to_answer = prev_answers[example_id]
                example = examples[example_id]
                prev_label_ids = example["prev_label_ids"]
                model_label_ids = np.zeros_like(prev_label_ids)
                for i in range(model_label_ids.shape[1]):
                    row_id = example["row_ids"][0, i].numpy() - 1
                    col_id = example["column_ids"][0, i].numpy() - 1
                    if row_id >= 0 and col_id >= 0 and example["segment_ids"][0, i].numpy() == 1:
                        model_label_ids[0, i] = int(
                            coords_to_answer[(col_id, row_id)])
                examples[example_id]["prev_label_ids"] = model_label_ids

        for example_id in examples:
            example = examples[example_id]
            result = get_outputs(model, example)
            results.append(result)
        all_results.extend(results)
        prev_answers = {}
        for prediction in results:
            question_id = prediction["question_id"][0,
                                                    0].numpy().decode("utf-8")
            table_id, annotator, _ = text_utils.parse_question_id(question_id)
            example_id = (table_id, annotator)
            example = examples[example_id]
            probabilities = prediction["probabilities"][0].numpy()

            # Compute average probability per cell, aggregating over tokens.
            coords_to_probs = collections.defaultdict(list)
            for i, p in enumerate(probabilities):
                segment_id = prediction["segment_ids"][0][i].numpy()
                col = prediction["column_ids"][0][i].numpy() - 1
                row = prediction["row_ids"][0][i].numpy() - 1
                if col >= 0 and row >= 0 and segment_id == 1:
                    coords_to_probs[(col, row)].append(p)

            coords_to_answer = {}
            for key in coords_to_probs:
                coords_to_answer[key] = np.array(
                    coords_to_probs[key]).mean() > 0.5
            prev_answers[example_id] = coords_to_answer

    return all_results


def convert_interactions_to_examples(converter, tables_and_queries, filename="test.tfrecord"):
    """Calls Tapas converter to convert interaction to example."""

    filename = os.path.join("temp", filename)
    for idx, (table, queries) in enumerate(tables_and_queries):
        interaction = interaction_pb2.Interaction()
        for position, query in enumerate(queries):
            question = interaction.questions.add()
            question.original_text = query
            question.id = f"{idx}-0_{position}"
        for header in table[0]:
            interaction.table.columns.add().text = header
        for line in table[1:]:
            row = interaction.table.rows.add()
            for cell in line:
                row.cells.add().text = cell
    number_annotation_utils.add_numeric_values(interaction)
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(interaction.questions)):
            try:
                writer.write(converter.convert(
                    interaction, i).SerializeToString())
            except ValueError as e:
                print(
                    f"Can't convert interaction: {interaction.id} error: {e}")
    return filename


def parser_fn(serialized_example):
    feature_types = {
        "input_ids":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH], tf.int64),
        "input_mask":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH], tf.int64),
        "segment_ids":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH], tf.int64),
        "column_ids":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH], tf.int64),
        "row_ids":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH], tf.int64),
        "prev_label_ids":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH],
                              tf.int64,
                              default_value=[0] * _MAX_SEQ_LENGTH),
        "column_ranks":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH], tf.int64),
        "inv_column_ranks":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH], tf.int64),
        "numeric_relations":
        tf.io.FixedLenFeature([_MAX_SEQ_LENGTH],
                              tf.int64,
                              default_value=[0] * _MAX_SEQ_LENGTH),
        "label_ids": tf.io.FixedLenFeature([_MAX_SEQ_LENGTH], tf.int64),
        "question_id": tf.io.FixedLenFeature([1], tf.string),
        "question_id_ints": tf.io.FixedLenFeature([text_utils.DEFAULT_INTS_LENGTH],
                                                  tf.int64,
                                                  default_value=[0] *
                                                  text_utils.DEFAULT_INTS_LENGTH)
    }
    example = tf.io.parse_single_example(serialized_example, feature_types)
    example_ = {}
    for k, v in example.items():
        if v.dtype == tf.int64:
            example_.update({k: tf.cast(v, tf.int32)})
        else:
            example_.update({k: v})
    return example_


def parse_coordinates(raw_coordinates):
    """Parses cell coordinates from text."""
    return {ast.literal_eval(x) for x in ast.literal_eval(raw_coordinates)}


class Model(object):

    def __init__(self, model_dir: Text, task: Text, tf_record_filename="test.tfrecord"):
        self.tf_record_filename = tf_record_filename
        self.task = task
        vocab_file = os.path.join(model_dir, "vocab.txt")
        bert_config_file = os.path.join(model_dir, "bert_config.json")
        classifier_conversion_config = tf_example_utils.ClassifierConversionConfig(vocab_file=vocab_file,
                                                                                   max_seq_length=_MAX_SEQ_LENGTH,
                                                                                   max_column_id=_MAX_TABLE_ID,
                                                                                   max_row_id=_MAX_TABLE_ID,
                                                                                   strip_column_names=False,
                                                                                   add_aggregation_candidates=False)
        self.converter = tf_example_utils.ToClassifierTensorflowExample(
            classifier_conversion_config)
        self.bert_config = json.load(open(bert_config_file))
        self.tapas_config = get_config(task, self.bert_config, model_dir)
        self.tapas = get_model(self.tapas_config, _MAX_SEQ_LENGTH)

    def __call__(self, table, queries):
        tables_and_queries = [(table, queries)]
        tfrecord = convert_interactions_to_examples(self.converter, tables_and_queries,
                                                    self.tf_record_filename)
        test_dataset = tf.data.TFRecordDataset(tfrecord)
        test_dataset = test_dataset.map(parser_fn)
        test_dataset = test_dataset.batch(1)
        test_examples = [example for example in test_dataset]
        if self.task.upper() == "SQA":
            results = compute_prediction_sequence(self.tapas, test_examples)
        else:
            results = [get_outputs(self.tapas, example)
                       for example in test_examples]
        predictions = prediction_utils.get_predictions(results, do_model_aggregation=self.tapas.do_model_aggregation,
                                                       do_model_classification=False,
                                                       cell_classification_threshold=_CELL_CLASSIFICATION_THRESHOLD)
        all_coordinates = []
        results = []
        for row in predictions:
            coordinates = parse_coordinates(
                row["answer_coordinates"])
            all_coordinates.append(coordinates)
            answers = ', '.join([table[row + 1][col]
                                    for row, col in coordinates])
            position = int(row['position'])
            results.append({"query": queries[position], "answer": answers, "answer_probablities":row["answer_probablities"]})
        return results
