"""

Author: Wissam Mammar kouadri
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from gensim import models
import numpy as np
from torch import nn
import torch
import pickle
import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
PROJECT_PATH = os.getcwd ()


# load training dataset from a  give path and return a formating dataset ready to use
# @:param
#       path: th path of the dataset type: string
#       PATH_W2V: path too the word embbeding type: string
#       label_type: the label formating type normal for normal encoding and binary for onehot type: string
# @:returns
#        X: encoded reviews type numpy.array
#        Y: encoded labels type numpy.array


def load_dataset(path  , label_type="Normal"):

    tokenizer = BertTokenizer.from_pretrained ('bert-base-uncased')

    # read the csv
    df_dataset = pd.read_csv (path , sep=";" , encoding="utf-8")

    # delete no ascii chars and tokenize
    df_dataset['Review'] = df_dataset["Review"].apply (
        lambda x: ''.join ([" " if ord (i) < 32 or ord (i) > 126 else i for i in x]))
    #padding

    #
    #df_dataset["Review"]=df_dataset["Review"].apply (lambda x: padding (x , max_length , "&"))
    #print(len(df_dataset["Review"].loc[0]))

    #mark the beguening and the end of the sentence
    df_dataset["Review"] = df_dataset["Review"].apply (lambda x: "[CLS] " + x + " [SEP]")
    df_dataset["Review"] = df_dataset["Review"].apply (lambda x: tokenizer.tokenize (x))
    max_length = get_max_length (df_dataset)
    df_dataset["Review"] = df_dataset["Review"].apply (lambda x: padding (x , max_length , "[PAD]"))
    #get id
    indexed_tokens = df_dataset["Review"].apply (lambda x: tokenizer.convert_tokens_to_ids (x))
    #tokenizer.convert_tokens_to_ids (tokenized_text)
    segments_ids = df_dataset["Review"].apply (lambda x: [1] * len (x))

    # formating labels
    df_dataset["Golden"] = label_formating (df_dataset , label_type=label_type)

    # return the encoded data and labels en numpy array
    return np.stack (indexed_tokens.values , axis=0) , np.stack (segments_ids.values , axis=0), np.stack (df_dataset["Golden"].values , axis=0)


# load  dataset for prediction from a give path and return a formating dataset ready to use
# @:param
#       path: the path of the dataset type: string
#       word2idx: codding dictionnary of the sentence type: dict
#       max_length: the max length of the string  type: int
# @:returns
#       X: encoded reviews type numpy.array


def load_dataset_predict(path, max_length ):
    # read the dataset from csv
    tokenizer = BertTokenizer.from_pretrained ('bert-base-uncased')

    # read the csv
    df_dataset = pd.read_csv (path , sep=";" , encoding="utf-8")

    # delete no ascii chars and tokenize
    df_dataset['Review'] = df_dataset["Review"].apply (
        lambda x: ''.join ([" " if ord (i) < 32 or ord (i) > 126 else i for i in x]))



    # mark the beguening and the end of the sentence
    df_dataset["Review"] = df_dataset["Review"].apply (lambda x: "[CLS] " + x + " [SEP]")

    #tokenize
    df_dataset["Review"] = df_dataset["Review"].apply (lambda x: tokenizer.tokenize (x))

    # padding
    df_dataset["Review"] = df_dataset["Review"].apply (lambda x: padding (x , max_length , "[PAD]"))

    # get id
    indexed_tokens = df_dataset["Review"].apply (lambda x: tokenizer.convert_tokens_to_ids (x))
    # tokenizer.convert_tokens_to_ids (tokenized_text)
    segments_ids = df_dataset["Review"].apply (lambda x: [1] * len (x))
    # return the encoded data and labels en numpy array
    return np.stack (indexed_tokens.values , axis=0) , np.stack (segments_ids.values , axis=0)

    # return the encoded data en numpy array


# formating label from positive neutral negative to label tyope
# @:param:
#       df_dataset: the dataset
#       label_type: the type of label encoding normal for simple classes and binary for onehote
# @:return
#       encoded labels
def label_formating(df_dataset , label_type="normal"):
    if label_type == "normal":

        df_dataset["Golden"] = df_dataset["Golden"].apply (
            lambda x: 2 if x == "Positive" else 0 if x == "Negative" else 1)

    else:

        if label_type == "binary":
            df_dataset["Golden"] = df_dataset["Golden"].apply (
                lambda x: [1 , 0 , 0] if x == "Positive" else [0 , 0 , 1] if x == "Negative" else [0 , 1 , 0])

    return df_dataset["Golden"]


# inverse the formating of labels from label type to positive neutral negative
def label_invers_formating(Y , label_type="normal"):
    if label_type == "normal":
        convert = (lambda Y: ["Positive" if x == 2 else "Negative" if x == 0 else "Neutral" for x in Y])
        return convert (Y)

    else:
        if label_type == "binary":
            to_int = lambda Y: [[1 if x >= 0.5 else 0 for x in z] for z in Y]
            Y = to_int (Y)
            convert = lambda Y: ["Positive" if x == [1 , 0 , 0] else "Negative" if x == [0 , 0 , 1] else "Neutral" for x
                                 in Y]
            return convert (Y)


# split dataset to training and test
def split_dataset(df_dataset):
    df_dataset.columns = ["Id" , "Review" , "Golden"]

    # shufll dataset
    df_dataset = df_dataset.sample (frac=1).reset_index (drop=True)

    # split the dataset
    X_train , X_test , Y_train , Y_test = train_test_split (df_dataset["Review"].values , df_dataset["Golden"].values)



    return X_train , X_test , Y_train , Y_test


def get_max_length(df_dataset ):
    # calculate max length sentece
    all_words = [word for tokens in df_dataset["Review"] for word in tokens]
    sentence_lengths = [len (tokens) for tokens in df_dataset["Review"]]

    # get training vocab
    TRAINING_VOCAB = sorted (list (set (all_words)))
    print ("%s words total, with a vocabulary size of %s" % (len (all_words) , len (TRAINING_VOCAB)))
    print ("Max sentence length is %s" % max (sentence_lengths))


    return   max (sentence_lengths)







# padding sentence (make them all with the same length)
def padding(sentence , max_length , pad_char):
    new_sentence = []

    for i in range ((max_length)):
        if i < len (sentence):
            new_sentence.append (sentence[i])
        else:
            new_sentence.append (pad_char)

    return (new_sentence)




def bert_embedding(text):

    tokenizer = BertTokenizer.from_pretrained ('bert-base-uncased')
    #text = "Here is the sentence I want embeddings for."
    marked_text = "[CLS] " + text + " [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize (marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids (tokenized_text)
    segments_ids = [1] * len (tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor ([indexed_tokens])
    segments_tensors = torch.tensor ([segments_ids])
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained ('bert-base-uncased')
    model.eval ()
    with torch.no_grad ():
        encoded_layers , _ = model (tokens_tensor , segments_tensors)
        token_embeddings = torch.stack (encoded_layers , dim=0)
        token_embeddings = torch.squeeze (token_embeddings , dim=1)
        token_embeddings = token_embeddings.permute (1 , 0 , 2)

        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum (token[-4:] , dim=0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append (sum_vec)

        print ('Shape is: %d x %d' % (len (token_vecs_sum) , len (token_vecs_sum[0])))

    print (segments_ids)


if __name__ == '__main__':
   load_dataset ( "../data/news.csv")

