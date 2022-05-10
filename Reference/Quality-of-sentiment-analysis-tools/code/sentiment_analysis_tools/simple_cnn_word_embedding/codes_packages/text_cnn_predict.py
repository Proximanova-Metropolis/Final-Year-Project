"""
Author: Wissam Mammar kouadri

"""

import text_cnn_utils
import pandas as pd
from sklearn.metrics  import accuracy_score
import argparse
from  torch import cuda
from torch.autograd import Variable
import torch
import pickle


use_cuda= cuda.is_available()


def predict (input_path,word2idx, save_path= "../model/weights0.pt"):
    #load the model
    model = (torch.load (save_path))
    model.eval ()

    #load and parse the data
    X = text_cnn_utils.load_dataset_predict (input_path , word2idx , model.input_len)
    X = X.astype (int)
    X = torch.from_numpy (X).long ()

    #prediction
    inputs = Variable (X)
    preds , vector = model (inputs)
    preds = torch.max (preds , 1)[1]
    pred_y = preds.data
    return pred_y


def main(input_path, out_path):
    with open ("../log/words" , 'rb') as f:
        word2idx = pickle.load (f)
    Y=predict(input_path,word2idx,save_path="../model/weight9.pt").numpy()

    Y = text_cnn_utils.label_invers_formating (( Y.tolist()) , label_type="normal")
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
    parser.add_argument ('--save_path' , type=str , default='../data/results' ,
                         help='path to save results')
    args= parser.parse_args ()
    main(args.input_path,args.out_path, save_path=args.save_path)