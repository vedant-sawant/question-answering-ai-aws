import torch
from model import BertQnA
from flask import request
import flask
import os
from flask import Flask, render_template, request


# Tokenizer / model
from transformers import DistilBertForQuestionAnswering

model = DistilBertForQuestionAnswering.from_pretrained("model/")
# Tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("model/")
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import model
import torch
from transformers import BertForQuestionAnswering
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", pred="Please ask a question!")


@app.route("/predict", methods=["POST"])
def predict():
    data = [request.form["question"]]

    name = [request.form["name"]]

    answer = model.BertQnA(name[0], data[0])

    return render_template(
        "index.html", pred=f"{name} ...I think the answer is {answer} !?"
    )


if __name__ == "__main__":

    app.run(debug=True)
