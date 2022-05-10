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


def get_acc(input_path_log ):
    df1= pd.read_csv(input_path_log, sep=";")
    return accuracy_score(df1["Pred"],df1["Golden"])


def main(args):
    datasets = ["AMAZON" , "NEWS" , "STANFORD"]
    embbedings = ["bert" , "char+\nG_news" , "Glove" , "G_news"]
    dirs=[join(args.log_path,d) for d in listdir(args.log_path)]
    #embbedings=[d for d in listdir(args.log_path)]
    fig = plt.figure (figsize=(1,2))
    nb_dir=0
    j=0
    dfi = pd.DataFrame (columns=["AMAZON" , "NEWS" , "STANFORD"], index=embbedings)
    for dir in dirs:
        means = []
        nb_dir=nb_dir +1
        onlyfiles=[f for f in listdir(dir) if isfile(join(dir,f)) ]
        i = 0
        for file in onlyfiles:
            #print("Acc",get_acc(join(dir, file)), file)
            print("Inc",get_acc(join(dir, file)), file)
            #ax = fig.add_subplot (1 , 6 , nb_dir)
            #print (column , df[column].mean ())
            dfi[datasets[i]][embbedings[nb_dir-1]] = get_acc(join(dir, file))
            i = i+1
            #means.append (get_inc_mean_inc(join(dir, file)))
            #ets[dataset] , df[column].mean () , color="black")

        #ax = fig.add_subplot (2 , 2 , nb_dir)
        #plt.bar (datasets , means , color="black")
        #ax.set_title (embbedings[nb_dir - 1] , fontsize=10)
    print(dfi)
    ax = fig.add_subplot (1, 1,  1)
    ax.set_title("Acc & Embedding Type", fontsize=8)
    bars = dfi.plot (kind='bar' , ax=ax , width=0.5 , colors=["#660000" , "#ff6666", "#ffb2b2"] , linewidth=0.5 ,
                     edgecolor="black", alpha=0.8)
    plt.xticks ([0,1,2,3] , embbedings , rotation='horizontal')

    mean_1=dfi.loc[embbedings[0]].mean ()
    mean_2 = dfi.loc[embbedings[1]].mean ()
    mean_3 = dfi.loc[embbedings[2]].mean ()
    mean_4= dfi.loc[embbedings[3]].mean ()
    print(mean_1)
    print (mean_2)
    print (mean_3)
    print (mean_4)
    #ax.text (10 , 1  , str (mean_1) , color='blue' , fontweight='bold')
    #ax.get_legend ().remove ()
    ax.set_ylabel ("Accuracy" , fontsize=8)
    ax.set_xlabel ("Embedding Type" , fontsize=8)
    plt.xticks (rotation=45 , fontsize=6)
    plt.yticks (rotation=45 , fontsize=8)
    ax.legend (loc="lower left" ,  prop={'size': 6})
    #ax = fig.add_subplot (1 , 2 , 2)
    #plt.axis ('off')
    #plt.text (0.05 , 1.05 , embbedings[0]+"_acc_mean:"+str(round(mean_1,2)) , horizontalalignment='center' , verticalalignment='top' , transform=ax.transAxes, fontsize=8)
    #plt.text (0.6 , 1.05 , embbedings[1] + "_acc_mean:" + str (round (mean_2 , 2)) , horizontalalignment='center' , verticalalignment='top' , transform=ax.transAxes,fontsize=8)
    #plt.text (0.05 , 1.15, embbedings[2] + "_acc_mean:" + str (round (mean_3 , 2)) , horizontalalignment='center' ,verticalalignment='top' , transform=ax.transAxes,fontsize=8)
    #plt.text (0.6 , 1.15 , embbedings[3] + "_acc_mean:" + str (round (mean_4 , 2)) , horizontalalignment='center' ,verticalalignment='top' , transform=ax.transAxes, fontsize=8)
    plt.subplots_adjust (hspace=0.5 , wspace=0.3)
    plt.legend (prop={'size': 5})

    plt.show ()
if __name__ =='__main__':
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--log_path' , type=str ,
                         default="D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/public_folder/experiments/logs/intool_inc_polar_fact" ,
                         help='input path of logs')
    args = parser.parse_args ()
    main(args)




