import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
import numpy as np
#Get files from exp directory
def intertool_inc(args):
    mypath = args.log_path
    # dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
    datasets=["AMAZONs","NEWS","STANFORD"]
    tools = ["CHAR_CNN" , "REC_NN"  , "SENTICNET" , "SENTIWORDNET", "TXT_CNN" , "VADER"]
    fig = plt.figure ()
    # sns.set (style="whitegrid")
    sns.set (font_scale=0.6)

    # sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))


    # for dir in dirs:
    # path_dir = join (mypath , dir)
    onlyfiles = [f for f in listdir (mypath) if isfile (join (mypath , f))]
    # nbdir = nbdir + 1
    numfile = 0
    dataset= 0
    for f in onlyfiles:
        nbdir = 0
        path_file = join (mypath , f)
        df = pd.read_csv (path_file , sep=";")
        df.columns = ["Id" , "CHAR_CNN" , "REC_NN" , "SENTIWORDNET" , "SENTICNET" , "TXT_CNN" , "VADER"]
        # print(df.columns)
        means = []


        for column in tools:
            nbdir = nbdir + 1
            #ax = fig.add_subplot (3 , 6 , numfile)
            #means.append (df[column].mean ())

            #sns.distplot (df[column] , ax=ax , color="black")
            ax = fig.add_subplot (1 , 6 , nbdir)
            print(column, df[column].mean () )
            means.append(df[column].mean () )
            plt.bar (datasets[dataset] , df[column].mean () , color="black")

            # sns.distplot (x=["Amazon","Stanford", "News"],Y=means , ax=ax , color="black")
            if dataset == 2:
                ax.set (xlabel='X= Inconsistency mean' , ylabel='Y= Nb clusters that have inconsistency X')
            # ax.grid (False)
        print(means)
        print("means", sum(means)/len(means))
        dataset=dataset+1

    plt.subplots_adjust (hspace=0.3 , wspace=0.3)
    plt.show ()
    fig.savefig (args.out_path)

if __name__ == '__main__':
    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--log_path' , type=str , default="./data/dataset.csv" ,
                         help='input path of logs')

    parser.add_argument ('--out_path' , type=str , default='./data/sentiment_dataset_sentic.csv' ,
                         help='output of plots')

    args = parser.parse_args ()
    intertool_inc(args)

