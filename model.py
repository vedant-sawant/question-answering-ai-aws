from transformers import BertTokenizer, BertForSequenceClassification
import torch

from transformers import DistilBertForQuestionAnswering

model = DistilBertForQuestionAnswering.from_pretrained("model/")
# Tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("model/")
import os

import numpy as np
import os

import torch


def BertQnA(answer_text, question):

    # tokenize

    encoded_dict = tokenizer.encode_plus(
        text=question, text_pair=answer_text, add_special_tokens=True
    )

    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = encoded_dict["input_ids"]

    # Report how long the input sequence is.
    print("Query has {:,} tokens.\n".format(len(input_ids)))

    # Segment Ids
    segment_ids = encoded_dict["token_type_ids"]

    # evaluate
    output = model(torch.tensor([input_ids]))

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(output["start_logits"])
    answer_end = torch.argmax(output["end_logits"])

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += " " + tokens[i]

    return answer
