# Author: Wissam  Mammar kouadri

"""

evaluate inconsistency  and accuracy following different learning hyperparameters

"""



import pandas as pd
import os
import argparse
import operator
from sentiment_analysis_tools.simple_cnn_word_embedding.codes_packages import text_cnn_train , text_cnn_predict , text_cnn_model
import numpy as np
from os.path import join
from os import listdir
from os.path import isfile , join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances , cosine_similarity
from sklearn.metrics import accuracy_score
import tempfile
from calculate_inconsistency import  intool_inconsistency

def run(input_path="" , log_output="" , dataset_split=0.75 , dropout=0.5 , learning_rate=0.0002 , training_dataset="" ,
        activation_function="softmax" ,
        nb_epochs=100 , batch_size=300 , path_w2v="" , folds=2 , model_path=""):
    test = ["../../../../data/clean_sentiment_datasets/amazon.csv" ,
            "../../../../data/clean_sentiment_datasets/news.csv" ,
            "../../../../data/clean_sentiment_datasets/stanford.csv"]
    text_cnn_train.main (training_dataset , path_w2v=path_w2v , nb_epoch=nb_epochs , folds=folds ,
                         save_path=model_path , batch_size=batch_size , lr=learning_rate , dropout_prob=dropout)
    acc = []
    inc = []

    # predict
    # main(input_path, out_path):

    for t in test:
        text_cnn_predict.main (t , log_output , save_path=join (model_path , str (nb_epochs - 1) + ".pt"))
        # calculate inconsistency

        intool_inconsistency (log_output , "temp")
        df_dataset = pd.read_csv (log_output , sep=";")
        df_inc = pd.read_csv ("temp" , sep=";")
        inc.append (df_inc["Inc"].mean ())

        # calculate accuracy
        acc.append (accuracy_score (df_dataset["Golden"] , df_dataset["Pred"]))
    # save parameters
    return acc , inc


def main(args):
    batch_size = [150 , 300 , 200 , 500 , 600 , 700 , 1000]
    nb_epochs = [50 , 100 , 150 , 200 , 300 , 3000]
    cv = [2 , 5 , 10 , 15]
    dropout = [0 , 0.2 , 0.3 , 0.5, 0.7]
    lr = [0.001 , 0.00015 , 0.0002 , 0.0003]

    input_path = args.input_path
    log_output = args.out_path
    path_w2v = args.path_w2v
    model_path = args.model_path
    training_dataset = args.training_dataset
    results_path = args.results_path
    training_datasets = ["../../../../data/sentiment_dataset_with_labels/news_headlines.csv" ,
                         "../../../../data/sentiment_dataset_with_labels/amazon_reviews_summary.csv" ,
                         "../../../../data/sentiment_dataset_with_labels/stanford-sentiment-treebank.test.csv"]

    with open (results_path , 'w') as out_file:

        for b in batch_size:
            model_path = join (model_path , str (b) + "batch")
            if os.path.exists (model_path) == False:
                os.mkdir (model_path)
            acc , inc = run (input_path=input_path , log_output=log_output , training_dataset=training_dataset ,
                             batch_size=b , path_w2v=path_w2v , model_path=model_path)
            out_file.write (
                "batch_size = " + str (b) + " acc amazon   " + str (acc[0]) + " inc amazon   " + str (inc[0]) + "\n")
            out_file.write (
                "batch_size = " + str (b) + " acc news     " + str (acc[1]) + " inc news     " + str (inc[1]) + "\n")
            out_file.write (
                "batch_size = " + str (b) + " acc stanford " + str (acc[2]) + " inc stanford " + str (inc[2]) + "\n")

        for e in nb_epochs:
            acc , inc = run (input_path=input_path , log_output=log_output , training_dataset=training_dataset ,
                             nb_epochs=e , path_w2v=path_w2v , model_path=model_path)
            out_file.write (
                "nb_epochs = " + str (e) + " acc amazon " + str (acc[0]) + " inc amazon " + str (inc[0]) + "\n")
            out_file.write ("nb_epochs = " + str (e) + " acc news " + str (acc[1]) + " inc news " + str (inc[1]) + "\n")
            out_file.write (
                "nb_epochs = " + str (e) + " acc stanford " + str (acc[2]) + " inc stanford " + str (inc[2]) + "\n")

        for c in cv:
            acc , inc = run (input_path=input_path , log_output=log_output , training_dataset=training_dataset ,
                             folds=c , path_w2v=path_w2v , model_path=model_path)
            out_file.write ("cv = " + str (c) + " acc amazon " + str (acc[0]) + " inc amazon " + str (inc[0]) + "\n")
            out_file.write ("cv = " + str (c) + " acc news " + str (acc[1]) + " inc news " + str (inc[1]) + "\n")
            out_file.write (
                "cv = " + str (c) + " acc stanford " + str (acc[2]) + " inc stanford " + str (inc[2]) + "\n")

        for d in dropout:
            acc , inc = run (input_path=input_path , log_output=log_output , training_dataset=training_dataset ,
                             dropout=d , path_w2v=path_w2v , model_path=model_path)
            out_file.write (
                "dropout = " + str (d) + " acc amazon " + str (acc[0]) + " inc amazon " + str (inc[0]) + "\n")
            out_file.write ("dropout = " + str (d) + " acc news " + str (acc[1]) + " inc news " + str (inc[1]) + "\n")
            out_file.write (
                "dropout = " + str (d) + " acc stanford " + str (acc[2]) + " inc stanford " + str (inc[2]) + "\n")

        for l in lr:
            acc , inc = run (input_path=input_path , log_output=log_output , training_dataset=training_dataset ,
                             learning_rate=l , path_w2v=path_w2v , model_path=model_path)
            out_file.write ("lr = " + str (l) + " acc amazon " + str (acc[0]) + " inc amazon " + str (inc[0]) + "\n")
            out_file.write ("lr = " + str (l) + " acc news " + str (acc[1]) + " inc news " + str (inc[1]) + "\n")
            out_file.write (
                "lr = " + str (l) + " acc stanford " + str (acc[2]) + " inc stanford " + str (inc[2]) + "\n")

        for t in training_datasets:
            acc , inc = run (input_path=input_path , log_output=log_output , training_dataset=t ,
                             path_w2v=path_w2v , model_path=model_path)
            out_file.write ("train = " + str (t) + " acc amazon " + str (acc[0]) + " inc amazon " + str (inc[0]) + "\n")
            out_file.write ("train = " + str (t) + " acc news " + str (acc[1]) + " inc news " + str (inc[1]) + "\n")
            out_file.write (
                "train = " + str (t) + " acc stanford " + str (acc[2]) + " inc stanford " + str (inc[2]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser ()

    parser.add_argument ('--input_path' , type=str ,
                         default="./VLDB_submission/experiments/logs/logs_dev" ,
                         help=' input data path ')
    parser.add_argument ('--out_path' , type=str ,
                         default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='data save path')
    parser.add_argument ('--folds' , type=int ,
                         default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='nb cross validation folds')
    parser.add_argument ('--dropout' , type=float , help='data save path')
    parser.add_argument ('--learning_rate' , type=float ,
                         default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='learning rate of the model')
    parser.add_argument ('--training_dataset' , type=str ,
                         default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='path to training dataset')

    parser.add_argument ('--activation_function' , type=str ,
                         default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='activation function : tan, softmax')
    parser.add_argument ('--nb_epochs' , type=int ,
                         default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='number of training epochs')

    parser.add_argument ('--batch_size' , type=int ,
                         default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='the batch size')

    parser.add_argument ('--path_w2v' , type=str ,
                         default='./VLDB_submission/experimen /logs/logs_dev' , help='data save path')
    parser.add_argument ('--model_path' , type=str , default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='path to the pre-traind model ')

    parser.add_argument ('--results_path' , type=str ,
                         default='../../../../experiments/logs/hyperparamater_inc/inc_acc.csv' , help='data save path')
    args = parser.parse_args ()
    main (args)


