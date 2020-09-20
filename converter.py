import json
import os
import shutil

import tensorflow as tf
from absl import app, flags

from classifier import _MAX_SEQ_LENGTH, get_config, get_model
from tapas.models.modeling import BertConfig

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None,
    "model_path for download models")

flags.DEFINE_string("save_path", None,
                    "save_path for saving converted weights")

flags.DEFINE_bool("do_reset", True,
                  "Select model type for weight conversion.\n"
                  "Reset refers to whether the parameter `reset_position_index_per_cell`"
                  "was set to true or false during training."
                  "In general it's recommended to set it to true")


flags.DEFINE_enum(
    "task", "SQA", ["SQA", "WTQ", "WIKISQL"], "task for converison")


def main(_):
    os.makedirs(FLAGS.save_path, exist_ok=True)
    bert_config_file = os.path.join(FLAGS.model_path,"bert_config.json")
    bert_config = json.load(open(bert_config_file))
    if FLAGS.do_reset:
        bert_config.update({"reset_position_index_per_cell":True})
    else:
        bert_config.update({"reset_position_index_per_cell":False})
    tapas_config = get_config(FLAGS.task, bert_config, "")
    tapas = get_model(tapas_config,_MAX_SEQ_LENGTH,False)
    dummy_input = [tf.ones((1,_MAX_SEQ_LENGTH))] * 10
    _ = tapas(*dummy_input,training=False)
    model_params = tapas.weights
    param_values = tf.keras.backend.batch_get_value(tapas.weights)
    tf_vars = tf.train.list_variables(os.path.join(FLAGS.model_path,"model.ckpt"))
    tf_vars_ = []
    for (name,size) in tf_vars:
        if name.endswith("adam_m") or name.endswith("adam_v") or name == 'global_step':
            continue
        else:
            tf_vars_.append((name,size))
    stock_values = {}
    for name, shape in tf_vars_:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(os.path.join(FLAGS.model_path,"model.ckpt"), name)
        stock_values.update({name:array})
    weight_map = {}
    weight_map['bert/embeddings/word_embeddings'] = 'bert_model/word_embeddings/embeddings:0'
    weight_map['bert/embeddings/token_type_embeddings_0'] = 'bert_model/embedding_postprocessor/segment_embeddings:0'
    weight_map['bert/embeddings/token_type_embeddings_1'] = 'bert_model/embedding_postprocessor/column_embeddings:0'
    weight_map['bert/embeddings/token_type_embeddings_2'] = 'bert_model/embedding_postprocessor/row_embeddings:0'
    weight_map['bert/embeddings/token_type_embeddings_3'] = 'bert_model/embedding_postprocessor/prev_label_embeddings:0'
    weight_map['bert/embeddings/token_type_embeddings_4'] = 'bert_model/embedding_postprocessor/column_ranks_embeddings:0'
    weight_map['bert/embeddings/token_type_embeddings_5'] = 'bert_model/embedding_postprocessor/inv_column_ranks_embeddings:0'
    weight_map['bert/embeddings/token_type_embeddings_6'] = 'bert_model/embedding_postprocessor/numeric_relations_embeddings:0'
    weight_map['bert/embeddings/position_embeddings'] = 'bert_model/embedding_postprocessor/position_embeddings:0'
    weight_map['bert/embeddings/LayerNorm/gamma'] = 'bert_model/embedding_postprocessor/layer_norm/gamma:0'
    weight_map['bert/embeddings/LayerNorm/beta'] = 'bert_model/embedding_postprocessor/layer_norm/beta:0'
    for i in range(bert_config["num_hidden_layers"]):
        weight_map[f'bert/encoder/layer_{i}/attention/self/query/kernel'] = f'bert_model/encoder/layer_{i}/self_attention/query/kernel:0'
        weight_map[f'bert/encoder/layer_{i}/attention/self/query/bias'] = f'bert_model/encoder/layer_{i}/self_attention/query/bias:0'
        weight_map[f'bert/encoder/layer_{i}/attention/self/key/kernel'] = f'bert_model/encoder/layer_{i}/self_attention/key/kernel:0'
        weight_map[f'bert/encoder/layer_{i}/attention/self/key/bias'] = f'bert_model/encoder/layer_{i}/self_attention/key/bias:0'
        weight_map[f'bert/encoder/layer_{i}/attention/self/value/kernel'] = f'bert_model/encoder/layer_{i}/self_attention/value/kernel:0'
        weight_map[f'bert/encoder/layer_{i}/attention/self/value/bias'] = f'bert_model/encoder/layer_{i}/self_attention/value/bias:0'
        weight_map[f'bert/encoder/layer_{i}/attention/output/dense/kernel'] = f'bert_model/encoder/layer_{i}/self_attention_output/kernel:0'
        weight_map[f'bert/encoder/layer_{i}/attention/output/dense/bias'] = f'bert_model/encoder/layer_{i}/self_attention_output/bias:0'
        weight_map[f'bert/encoder/layer_{i}/attention/output/LayerNorm/gamma'] = f'bert_model/encoder/layer_{i}/self_attention_layer_norm/gamma:0'
        weight_map[f'bert/encoder/layer_{i}/attention/output/LayerNorm/beta'] = f'bert_model/encoder/layer_{i}/self_attention_layer_norm/beta:0'
        weight_map[f'bert/encoder/layer_{i}/intermediate/dense/kernel'] = f'bert_model/encoder/layer_{i}/intermediate/kernel:0'
        weight_map[f'bert/encoder/layer_{i}/intermediate/dense/bias'] = f'bert_model/encoder/layer_{i}/intermediate/bias:0'
        weight_map[f'bert/encoder/layer_{i}/output/dense/kernel'] = f'bert_model/encoder/layer_{i}/output/kernel:0'
        weight_map[f'bert/encoder/layer_{i}/output/dense/bias'] = f'bert_model/encoder/layer_{i}/output/bias:0'
        weight_map[f'bert/encoder/layer_{i}/output/LayerNorm/gamma'] = f'bert_model/encoder/layer_{i}/output_layer_norm/gamma:0'
        weight_map[f'bert/encoder/layer_{i}/output/LayerNorm/beta']= f'bert_model/encoder/layer_{i}/output_layer_norm/beta:0'
    weight_map['bert/pooler/dense/kernel'] = 'bert_model/pooler_transform/kernel:0'
    weight_map['bert/pooler/dense/bias'] = 'bert_model/pooler_transform/bias:0'
    if FLAGS.task in ["SQA","WTQ"]:
        weight_map['column_output_weights'] = 'column_output_weights:0'
        weight_map['column_output_bias'] = 'column_output_bias:0'
    if FLAGS.task in ["WTQ","WIKISQL"]:
        weight_map['output_weights_agg'] = 'output_weights_agg:0'
        weight_map['output_bias_agg'] = 'output_bias_agg:0'
    weight_map['output_weights'] = 'output_weights:0'
    weight_map['output_bias'] = 'output_bias:0'
    weight_map = {v:k for k,v in weight_map.items()}
    loaded_weights = set()
    weight_value_tuples = []
    model_params = tapas.weights
    param_values = tf.keras.backend.batch_get_value(tapas.weights)
    skipped_weight_value_tuples = []
    for ndx, (param_value, param) in enumerate(zip(param_values, model_params)):
        stock_name = weight_map[param.name]

        if stock_name in stock_values:
            ckpt_value = stock_values[stock_name]

            if param_value.shape != ckpt_value.shape:
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                    "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(
                param.name, stock_name, FLAGS.model_path))
            skip_count += 1
    tf.keras.backend.batch_set_value(weight_value_tuples)

    tapas.save_weights(os.path.join(FLAGS.save_path,"model"))
    json.dump(bert_config,open(os.path.join(FLAGS.save_path,"bert_config.json"),'w'),indent=4)
    shutil.copyfile(os.path.join(FLAGS.model_path,"vocab.txt"),os.path.join(FLAGS.save_path,"vocab.txt"))


if __name__ == "__main__":
    app.run(main)
