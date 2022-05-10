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


def predict (input_path, save_path= "../model/weights0.pt"):
    #load the model
    model = (torch.load (save_path))
    model.eval ()

    #load and parse the data
    x1,x2 = text_cnn_utils.load_dataset_predict (input_path , model.input_len)
    x1 = x1.astype (int)
    x2 = x2.astype (int)
    x1 = torch.from_numpy (x1).long ()
    x2 = torch.from_numpy (x2).long ()

    #prediction
    x1 = Variable (x1)
    x2 = Variable (x2)
    preds , vector = model ([x1,x2])
    preds = torch.max (preds , 1)[1]
    pred_y = preds.data
    return pred_y


def main(input_path, out_path, save_path="../model/weights0.pt"):
    Y=predict(input_path,save_path="../model/weight9.pt").numpy()

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
    args= parser.parse_args ()
    main(args.input_path,args.out_path,save_path= args.save_path)