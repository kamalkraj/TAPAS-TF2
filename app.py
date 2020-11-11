import gradio as gr
import pandas as pd
from absl import app, flags

from classifier import Model

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None,
    "model_path for loading weights")

flags.DEFINE_enum(
    "task", "SQA", ["SQA", "WTQ", "WIKISQL"], "task for prediction")

model = None

def predict(file_obj,Question):
    global model
    df = pd.read_csv(file_obj.name,dtype=str)
    array = [df.columns]
    values = df.values
    array.extend(values)
    Question = [Question]
    output = model(array,Question)
    for out in output:
        out['answer_probablities'] = [float(x)*100 for x in out['answer_probablities']]
    return df, output

def main(_):
    global model
    model = Model(FLAGS.model_path,FLAGS.task)
    io = gr.Interface(predict, ["file",gr.inputs.Textbox(placeholder="Enter Question here...")], ["dataframe","json"])
    io.launch()

if __name__ == "__main__":
    app.run(main)
