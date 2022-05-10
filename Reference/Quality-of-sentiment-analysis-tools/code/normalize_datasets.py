"""
Author: Wissam  Mammar kouadri
"""

import pandas as pd
import os
import argparse

#stanford vider
def normalize(args):
        input_file=args.input_file
        out_file= args.out_file
        df=pd.read_csv(input_file,sep=";")
        df.columns = ["Id" , "Review" , "Golden" , "Pred"]
        input_file=input_file.lower()
        if("stanforddata" in input_file):
            df["Golden"]=  df['Golden'].apply(lambda y:  "Positive" if y > 0.6 else "Negative" if y <= 0.4 else "Neutral")
        if("news" in input_file):
            df["Golden"] = df['Golden'].apply (lambda y: "Positive" if y > 0 else "Negative" if y < 0 else "Neutral")
        if ("amazon" in input_file):
            df["Golden"] = df['Golden'].apply (lambda y: "Positive" if y > 3 else "Negative" if y < 3 else "Neutral")

        if("vader" in input_file):

            df["Pred"]=  df['Pred'].apply(lambda y:  "Positive" if y== "pos" else "Negative" if y == "neg" else "Neutral")

        if ("sentic" in input_file):
            df["Pred"] = df['Pred'].apply (lambda y: "Positive" if y == "POSITIVE" else "Negative" if y == "NEGATIVE" else "Neutral")


        if ("sentiwordnet" in input_file):
            df["Pred"] = df['Pred'].apply (lambda y: "Positive" if y == "positive" else "Negative" if y == "negative" else "Neutral")

        if ("rec_nn" in input_file):
            df["Pred"] = df['Pred'].apply (lambda y: "Positive" if y == "Very positive" else "Negative" if y == "Very negative" else y)

        if ("charcnn" in input_file):
            df["Pred"] = df['Pred'].apply (lambda y: "Positive" if y >=0.6 else "Negative" if y <0.5 else "Neutral")
            #df.to_csv (out_file , sep=";" , index=False)
        if ("cnn_text" in input_file):
            df["Pred"] = df['Pred'].apply (lambda y: "Positive" if y ==1 else "Negative" if y ==0  else "Neutral")
        df.to_csv (out_file , sep=";" , index=False)






if __name__ == '__main__':
    # Ignore warning message by tensor flow

    # model args
    parser = argparse.ArgumentParser (description='Normalize datasets')

    parser.add_argument ('--input_file' , type=str , default="../data/sentiment_datasets_augmented"
                                                             ,
                                                              help=' load path')
    parser.add_argument ('--out_file' , type=str , default='../data/sentiment_datasets_labels' ,
                         help='data save path')


    args = parser.parse_args()
    normalize(args)
