import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
from  sklearn.metrics import accuracy_score
#Get files from exp directory
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

from matplotlib.ticker import PercentFormatter
current_palette_7 = sns.color_palette("hls", 7)
sns.set_palette("Set2")


def get_acc(input_path_log ):
    df1= pd.read_csv(input_path_log, sep=";")
    return accuracy_score(df1["Pred"],df1["Golden"])


def main(args):

    datasets = ["AMAZON" , "NEWS" , "STANFORD"]

    batch_size_inc=[]
    #tools = ["CHAR" , "GLOVE" , "GOOGLE_NEWS" , "BERT" , "TXT_CNN" , "VADER"]
    #dirs=[join(args.log_path,d) for d in listdir(args.log_path)]
    #embbedings=[d for d in listdir(args.log_path)]
    fig = plt.figure (figsize=(1, 2))
    nb_dir=0
    j=0
    #dfi = pd.DataFrame (columns=["AMAZON" , "NEWS" , "STANFORD"], index=embbedings)
    #for dir in dirs:
    #    means = []
    #    nb_dir=nb_dir +1
    onlyfiles=[join(args.log_path,f) for f in listdir(args.log_path) if isfile(join(args.log_path,f)) ]
    i = 0
    for file in onlyfiles:

            markers = ["^" , "o" , "s"]
            df=pd.read_csv(file, sep=";")
            sizes=df["lr"].values
            #accs=df["acc"].values
            #print(accs)
            incs=df["inc"].values
            ax = fig.add_subplot (1 ,1 , 1)
            sns.lineplot (sizes,incs  , ax=ax , markersize=6,marker=markers[i],linewidth=1,markerfacecolor='white',
         markeredgecolor='black',
         markeredgewidth=0.5)
            ax.spines['right'].set_visible (False)
            ax.spines['top'].set_visible (False)
            plt.xticks ([0.0001,0.00015,0.0002,0.0003],["1e-3","1.5e-3","2e-3","3e-3"],fontsize=8,rotation =45)
            plt.yticks (rotation=45,fontsize=8)
            #ax = fig.add_subplot (1 , 2 , 2)
            #sns.lineplot (sizes,accs  , ax=ax , markersize=6,marker=markers[i], linewidth=1,markerfacecolor='white', markeredgecolor='black', markeredgewidth=0.5)
            ax.set_xticks (sizes)
            ax.spines['right'].set_visible (False)
            ax.spines['top'].set_visible (False)

            #plt.xticks ( rotation=45,fontsize=8)
            plt.yticks (rotation=45,fontsize=8)

            i=i+1
            #means.append (get_inc_mean_inc(join(dir, file)))
            #ets[dataset] , df[column].mean () , color="black")

        #ax = fig.add_subplot (2 , 2 , nb_dir)
        #plt.bar (datasets , means , color="black")
        #ax.set_title (embbedings[nb_dir - 1] , fontsize=10)

    #ax.text (10 , 1  , str (mean_1) , color='blue' , fontweight='bold')
    #ax.get_legend ().remove ()
    ax = fig.add_subplot (1 , 1, 1)
    ax.set_ylabel ("Inconsistency_mean" , fontsize=8)
    ax.set_xlabel ("learning rate" , fontsize=8)
    ax.set_title("Inc & learnig rate",fontsize=8)
    import matplotlib.ticker as mtick

    ax.xaxis.set_major_formatter (mtick.FormatStrFormatter ('%.1e'))

    #ax = fig.add_subplot (1 , 2 , 2)
    #ax.set_ylabel ("Accuracy" , fontsize=8)
    #ax.set_xlabel ("batch_size" , fontsize=8)
    #plt.figlegend (datasets , loc='center left' , ncol=3 , labelspacing=0., prop={'size': 5})
    #ax.set_title("Acc and batch size",fontsize=8)


    #ax.legend (loc="lower left" ,  prop={'size': 6})
    #ax = fig.add_subplot (1 , 2 , 2)
    #plt.axis ('off')
    plt.show ()
if __name__ =='__main__':
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--log_path' , type=str ,
                         default="D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/public_folder/experiments/logs/intool_inc_polar_fact" ,
                         help='input path of logs')
    args = parser.parse_args ()
    main(args)




