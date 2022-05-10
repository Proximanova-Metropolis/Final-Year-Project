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
sns.set_palette(current_palette_7)

def get_inc_mean_inc(input_path_log ):
    df1= pd.read_csv(input_path_log, sep=";")
    return df1["Inc"].mean()

def get_acc(input_path_log ):
    df1= pd.read_csv(input_path_log, sep=";")
    return accuracy_score(df1["Pred"],df1["Golden"])


def main(args):
    datasets = ["AMAZON" , "NEWS" , "STANFORD"]
    #tools = ["CHAR" , "GLOVE" , "GOOGLE_NEWS" , "BERT" , "TXT_CNN" , "VADER"]
    fig = plt.figure (figsize=(1,2))

    nb_dir=0
    j=0
    #dfi_inc = pd.DataFrame (columns=["AMAZON" , "NEWS" , "STANFORD"])
    dfi_acc = pd.DataFrame (columns=["AMAZON" , "NEWS" , "STANFORD"])
    #for dir in dirs:
    #    means = []
    #    nb_dir=nb_dir +1
    onlyfiles=[join(args.log_path,f) for f in listdir(args.log_path) if isfile(join(args.log_path,f)) ]
    i = 0
    for file in onlyfiles:
        #print("Acc",get_acc(join(dir, file)), file)
        #print("Inc",get_inc_mean_inc(join(dir, file)), file)
        #ax = fig.add_subplot (1 , 6 , nb_dir)
        #print (column , df[column].mean ())
        df=pd.read_csv(file,sep=";")
        #dfi_inc[datasets[i]] = df["inc"]
        dfi_acc.loc[i] = df["acc"].to_numpy()
        i = i+1
        #means.append (get_inc_mean_inc(join(dir, file)))
        #ets[dataset] , df[column].mean () , color="black")

        #ax = fig.add_subplot (2 , 2 , nb_dir)
        #plt.bar (datasets , means , color="black")
        #ax.set_title (embbedings[nb_dir - 1] , fontsize=10)
    #print(dfi_inc)
    #print (dfi_acc)
    ax = fig.add_subplot (1, 1,  1)
    #ax.set_title("Inc & Training dataset", fontsize=8)
    #bars = dfi_inc.plot (kind='bar' , ax=ax , width=0.5 , colors=["#660000" , "#ff6666", "#ffb2b2"] , linewidth=0.5 , edgecolor="black", alpha=0.8)
    plt.xticks ([0,1,2,3] , datasets , rotation=45, fontsize=6)
    #ax.set_ylabel ("inconsistency_mean" , fontsize=8)
    #ax.set_xlabel ("Training dataset" , fontsize=8)
    plt.legend (prop={'size': 3})
    ax = fig.add_subplot (1 , 1 , 1)

    ax.set_title ("Acc & Training dataset" ,fontsize=8)
    bars = dfi_acc.plot (kind='bar' , ax=ax , width=0.5 , colors=["#660000" , "#ff6666" , "#ffb2b2"] , linewidth=0.5 ,edgecolor="black" , alpha=0.8)
    plt.xticks ([0 , 1 , 2 , 3] , datasets , rotation=45, fontsize=6)
    plt.yticks (rotation=45 , fontsize=6)
    mean_1=dfi_acc.iloc[0].mean ()
    mean_2 = dfi_acc.iloc[1].mean ()
    mean_4= dfi_acc.iloc[2].mean ()
    print("mean_1", mean_1)
    print ("mean_2" , mean_2)
    print ("mean_4" , mean_4)
    #ax.text (10 , 1  , str (mean_1) , color='blue' , fontweight='bold')
    ax.get_legend ().remove ()
    ax.set_ylabel ("Accuracy " , fontsize=8)
    ax.set_xlabel ("Training dataset" , fontsize=8)
    #ax = fig.add_subplot (1 , 2 , 2)
    #plt.axis ('off')
    #plt.text (0.05 , 1.05 , embbedings[0]+"_inc_mean:"+str(round(mean_1,2)) , horizontalalignment='center' ,verticalalignment='top' , transform=ax.transAxes, fontsize=8)
    #plt.text (0.35 , 1.05 , embbedings[1] + "_inc_mean:" + str (round (mean_2 , 2)) , horizontalalignment='center' ,verticalalignment='top' , transform=ax.transAxes,fontsize=8)
    #plt.text (0.65 , 1.05, embbedings[2] + "_inc_mean:" + str (round (mean_3 , 2)) , horizontalalignment='center' , verticalalignment='top' , transform=ax.transAxes,fontsize=8)
    #plt.text (0.9 , 1.05 , embbedings[3] + "_inc_mean:" + str (round (mean_4 , 2)) , horizontalalignment='center' ,verticalalignment='top' , transform=ax.transAxes, fontsize=8)
    plt.subplots_adjust (hspace=0.5 , wspace=0.3)
    plt.show ()
if __name__ =='__main__':
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--log_path' , type=str ,
                         default="D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/public_folder/experiments/logs/intool_inc_polar_fact" ,
                         help='input path of logs')
    args = parser.parse_args ()
    main(args)




