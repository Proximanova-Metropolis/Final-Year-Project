"""

Author: Wissam MAMMAR KOUADRI
"""

import char_cnn_utils
import pandas as pd
from sklearn.metrics  import accuracy_score
import argparse
from  torch import cuda
from torch.autograd import Variable
import torch
import pickle
import os

use_cuda= cuda.is_available()


def predict (input_path,word2idx,char2idx, save_path= "../model/weights9.pt"):
    #load the model
    model = (torch.load (save_path))
    model.eval ()
    #def load_dataset_predict(path , word2idx , max_length,char2idx,max_len_word):

    #load and parse the data
    X , X_CHAR= char_cnn_utils.load_dataset_predict (input_path , word2idx ,char2idx, model.input_len, model.char_input_len)
    X = X.astype (int)
    X_CHAR=X_CHAR.astype (int)
    X = torch.from_numpy (X).long ()
    X_CHAR=torch.from_numpy (X_CHAR).long ()

    #prediction
    input_word, input_char = Variable (X),Variable(X_CHAR)
    preds , vector = model ([input_word,input_char])
    preds = torch.max (preds , 1)[1]
    pred_y = preds.data
    return pred_y


def main(input_path, out_path):
    d = os.path.dirname(os.getcwd())
    print(d)
    with open (os.path.join(f"{d}", "Quality-of-sentiment-analysis-tools/code/sentiment_analysis_tools/char_embedding_cnn/log/words") , 'rb') as f:
        word2idx = pickle.load (f)
    with open (os.path.join(f"{d}", "Quality-of-sentiment-analysis-tools/code/sentiment_analysis_tools/char_embedding_cnn/log/chars") , 'rb') as f:
        char2idx = pickle.load (f)
    #def predict (input_path,word2idx,char2idx, save_path= "../model/weights9.pt"):

    Y=predict(input_path,word2idx,char2idx, save_path="../model/9.pt").numpy()

    Y = char_cnn_utils.label_invers_formating (( Y.tolist()) , label_type="normal")
    df=pd.read_csv(input_path, sep=";")
    truth =df["Golden"].values
    print('accuracy : {:.3f}' .format (accuracy_score(truth,Y)))
    df["Pred"]=Y
    df.to_csv(out_path, sep=";")


if __name__=="__main__":
    parser = argparse.ArgumentParser (description='Predict polarity with Convolutional Neural Networks forpolarity extraction in pytorch')
    parser.add_argument ('--input_path' , type=str , default='../data/augmeted_dataset/news.csv' ,
                         help='path to data file, it should be a csv file')
    parser.add_argument ('--out_path' , type=str , default='../data/results' ,
                         help='path to save results')

    args= parser.parse_args ()
    main(args.input_path,args.out_path)