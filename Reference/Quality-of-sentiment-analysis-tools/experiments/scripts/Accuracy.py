import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
#Get files from exp directory



def intool_inc(args):
    fig = plt.figure (figsize=(1 , 2))
    mypath= args.log_path

    fig = plt.figure ()
    #sns.set (style="whitegrid")

    #sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
    nbdir=0



    df= pd.read_csv(args.log_path,sep=";")
    means=df["Accuracy"]
    ax = fig.add_subplot (1 , 1 , 1)
    plt.bar (df["Method"] , means, width=0.5  , linewidth=0.5 ,
                     edgecolor="black", alpha=0.8)
    ax.set (xlabel='X= Methods' , ylabel='Y= Accuracy')

    ax.set_title("Inconsistency resolution results on news heads dataset")










    plt.subplots_adjust (hspace=0.3,wspace = 0.3)
    plt.show()
    fig.savefig (args.out_path)

if __name__ == '__main__':
    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--log_path' , type=str , default="D:/Users/wissam/Documents/These/these/papers_material/AAAI_submission/Data/accuracy_results_news.csv" ,
                         help='input path of logs')

    parser.add_argument ('--out_path' , type=str , default='./data/sentiment_dataset_sentic.csv' ,
                         help='output of plots')

    args = parser.parse_args ()
    intool_inc(args)

#gg(gg.mtcars, gg.aes( y=df["Inc"].values)) + gg.geom_boxplot()