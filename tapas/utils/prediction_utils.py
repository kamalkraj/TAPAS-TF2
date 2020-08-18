# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Writes prediction to a csv file."""

import collections
import copy
import csv
from typing import Iterable, Mapping, Text, Tuple

import numpy as np
import tensorflow as tf
from absl import logging

from tapas.utils import text_utils


def _get_question_id(features):
    """Restores question id from int sequence."""
    if "question_id_ints" in features:
        question_id = text_utils.ints_to_str(
            features["question_id_ints"].numpy()[0])
        if question_id:
            return question_id
    # TODO Remove once the data has been updated.
    return features["question_id"][0,0].numpy().decode("utf-8")


def get_cell_token_probs(prediction):
    probabilities = prediction["probabilities"][0].numpy()
    for i, p in enumerate(probabilities):
        segment_id = prediction["segment_ids"][0][i].numpy()
        col = prediction["column_ids"][0][i].numpy() - 1
        row = prediction["row_ids"][0][i].numpy() - 1
        if col >= 0 and row >= 0 and segment_id == 1:
            yield i, p


def get_mean_cell_probs(prediction):
    """Computes average probability per cell, aggregating over tokens."""

    coords_to_probs = collections.defaultdict(list)
    for i, prob in get_cell_token_probs(prediction):
        col = prediction["column_ids"][0][i].numpy() - 1
        row = prediction["row_ids"][0][i].numpy() - 1
        coords_to_probs[(col, row)].append(prob)
    return {
        coords: np.array(cell_probs).mean()
        for coords, cell_probs in coords_to_probs.items()
    }


def get_answer_indexes(
    prediction,
    cell_classification_threshold,
):
    """Computes answer indexes."""
    input_ids = prediction["input_ids"][0].numpy()

    span_indexes = prediction.get("span_indexes")
    span_logits = prediction.get("span_logits")
    if span_indexes is not None and span_logits is not None:
        best_logit, best_span = max(zip(span_logits, span_indexes.tolist()),)
        logging.log_every_n(
            logging.INFO,
            "best_span: %s, score: %s",
            500,
            best_span,
            best_logit,
        )
        return [input_ids[i] for i in range(best_span[0], best_span[1] + 1)]

    answers = []
    for i, prob in get_cell_token_probs(prediction):
        if prob > cell_classification_threshold:
            answers.append(input_ids[i])
    return answers


def get_predictions(
    predictions,
    do_model_aggregation,
    do_model_classification,
    cell_classification_threshold,
):
    """Writes predictions to an output TSV file.

    Predictions header: [id, annotator, position, answer_coordinates, gold_aggr,
    pred_aggr]

    Args:
      predictions: model predictions
      do_model_aggregation: Indicates whther to write predicted aggregations.
      do_model_classification: Indicates whther to write predicted classes.
      cell_classification_threshold: Threshold for selecting a cell.
    """
    results = []
    header = [
        "question_id",
        "id",
        "annotator",
        "position",
        "answer_coordinates",
        "answer",
    ]
    if do_model_aggregation:
        header.extend(["gold_aggr", "pred_aggr"])
    if do_model_classification:
        header.extend(["gold_cls", "pred_cls", "logits_cls"])

    for prediction in predictions:
        question_id = _get_question_id(prediction)
        max_width = prediction["column_ids"][0].numpy().max()
        max_height = prediction["row_ids"][0].numpy().max()

        if (max_width == 0 and max_height == 0 and
                question_id == text_utils.get_padded_question_id()):
            logging.info("Removing padded example: %s", question_id)
            continue

        cell_coords_to_prob = get_mean_cell_probs(prediction)

        answer_indexes = get_answer_indexes(
            prediction,
            cell_classification_threshold,
        )

        # Select the answers above a classification threshold.
        answer_coordinates = []
        answer_probablities = []
        for col in range(max_width):
            for row in range(max_height):
                cell_prob = cell_coords_to_prob.get((col, row), None)
                if cell_prob is not None:
                    if cell_prob > cell_classification_threshold:
                        answer_coordinates.append(str((row, col)))
                        answer_probablities.append(cell_prob)

        try:
            example_id, annotator, position = text_utils.parse_question_id(
                question_id)
            position = str(position)
        except ValueError:
            example_id = "_"
            annotator = "_"
            position = "_"
        prediction_to_write = {
            "question_id": question_id,
            "id": example_id,
            "annotator": annotator,
            "position": position,
            "answer_coordinates": str(answer_coordinates),
            "answer": str(answer_indexes),
            "answer_probablities": answer_probablities if len(answer_probablities) else [0.0]
        }
        if do_model_aggregation:
            prediction_to_write["gold_aggr"] = str(
                prediction["gold_aggr"][0][0].numpy())
            prediction_to_write["pred_aggr"] = str(prediction["pred_aggr"][0][0].numpy())
        if do_model_classification:
            prediction_to_write["gold_cls"] = str(
                prediction["gold_cls"][0])
            prediction_to_write["pred_cls"] = str(prediction["pred_cls"])
            prediction_to_write["logits_cls"] = str(
                prediction["logits_cls"])
        results.append(prediction_to_write)
    return results