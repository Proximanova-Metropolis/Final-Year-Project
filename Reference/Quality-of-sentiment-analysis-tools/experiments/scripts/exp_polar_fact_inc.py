import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
#Get files from exp directory
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

from matplotlib.ticker import PercentFormatter



def polar_fact_inc(args):
    path_fact= args.log_path_fact
    path_opinion= args.log_path_opinion
    dirs_fact = [dI for dI in listdir (path_fact) if isdir (join (path_fact , dI))]
    dirs_opinion= dirs = [dI for dI in listdir (path_opinion) if isdir (join (path_opinion , dI))]
    datasets=["AMAZON_PRODUCTS","NEWS_HEADLINES","STANFORD_TREEBANK"]
    tools = ["CHAR_CNN" , "CNN_TEXT" , "SENTICNET" , "SENTIWORDNET" , "STANFORD" , "VADER"]

    fig = plt.figure ()


    #sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))

    num_plot=1

    for i in range(len(dirs)):
        path_dirs_fact= join(path_fact,dirs_fact[i])
        path_dirs_opinion = join (path_opinion,dirs_opinion[i])

        onlyfiles_fact = [f for f in listdir(path_dirs_fact) if isfile(join(path_dirs_fact, f))]
        onlyfiles_opinion = [f for f in listdir(path_dirs_opinion) if isfile(join(path_dirs_opinion, f))]
        means_opinion=[]
        means_fact = []

        for j in range (len (onlyfiles_fact)):
            path_file_fact=join(path_dirs_fact,onlyfiles_fact[j])
            path_file_opinion = join (path_dirs_opinion , onlyfiles_opinion[j])
            df_fact= pd.read_csv(path_file_fact,sep=";")
            df_opinion = pd.read_csv (path_file_opinion , sep=";")
            #print(df.columns)
            means_opinion.append (df_opinion["Inc"].mean ())
            means_fact.append (df_fact["Inc"].mean ())
            #ax = fig.add_subplot (3 , 6 , num_plot)

            #sns.distplot (df_fact["Inc"],ax=ax)
            #sns.distplot (df_opinion["Inc"] , ax=ax)
            #plt.hist(df_fact["Inc"],density=1)
            #bins = np.linspace (0 , 1 , 8)
            #plt.hist ([df_fact["Inc"],df_opinion["Inc"] ],bins, density=1, label=["fact","opinion"])
            hist , bins = np.histogram (df_fact["Inc"])
            """
            ax.bar (bins[:-1] , hist.astype (np.float32) / hist.sum () , alpha=0.4,width=(bins[1] - bins[0]) ,label="fact",color = "r")
            hist , bins = np.histogram (df_opinion["Inc"])
            ax.bar (bins[:-1] , hist.astype (np.float32) / hist.sum () ,alpha=0.4, width=(bins[1] - bins[0]) ,label="opinion",color = "b")
            plt.legend (loc='upper right')"""

            colors_list = ['#5cb85c' , '#5bc0de' , '#d9534f']

            # Change this line to plot percentages instead of absolute values                                                              color=colors_list , edgecolor=None)


            #sns.barplot ( df_fact["Inc"], estimator=lambda x: len (x) / len (df_fact["Inc"]) ,ax=ax)
            #sns.barplot (df_opinion["Inc"] , estimator=lambda x: len (x) / len (df_opinion["Inc"]) , ax=ax)
            """
            hist , bins = np.histogram (df_opinion["Inc"])
            ax.bar (bins[:-1], hist.astype (np.float32) / hist.sum () , width=(bins[1] - bins[0]) )
            hist1 , bins1 = np.histogram (df_fact["Inc"])
            ax.bar (bins1[:-1] , hist1.astype (np.float32) / hist1.sum () , width=(bins1[1] - bins1[0]) )"""
            """
            ax.set_xlabel ('X= Inconsistency degree', fontsize=6)
            ax.set_ylabel('Y= proportion of clusters with inconsistency X', fontsize=6)
            ax.grid (False)

            if num_plot<=6 :
                ax.set_title (dirs[num_plot-1],fontsize=10)

            if num_plot%6 ==1:


                ax.set_ylabel (datasets[num_plot//6],fontsize=10)
                ax.yaxis.set_label_position ("left")

            num_plot = num_plot + 1
            #ax.tick_params (axis='both' , which='major' , pad=5)
            #ax.set_xlabel('X= Inconsistency degree',fontsize=10)

            """
            #ax.yaxis.set_label_position ("right")


#df_list.append(df["Inc"])"""

        """
        #df= pd.read_csv("D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/experiments/logs/Inconsistency_degree/CHARCNN/CHARCNNTEXTCLASSIFICATION_amazon_hight_quality_analysis_inc.csv",sep=";")
        df_new=pd.DataFrame(columns=["Amazon","Stanford","News"])
        df= pd.read_csv("D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/experiments/logs/Inconsistency_degree/CHARCNN/CHARCNNTEXTCLASSIFICATION_amazon_hight_quality_analysis_inc.csv",sep=";")
        df2= pd.read_csv("D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/experiments/logs/Inconsistency_degree/CHARCNN/CHARCNNTEXTCLASSIFICATION_news_hight_quality_analysis_inc.csv",sep=";")
        df3= pd.read_csv("D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/experiments/logs/Inconsistency_degree/CHARCNN/CHARCNNTEXTCLASSIFICATION_stanford_hight_quality_analysis_inc.csv",sep=";")
        df_new["Amazon"]=df["Inc"]
        df_new["News"]=df2["Inc"]
        df_new["Stanford"]=df3["Inc"]
        print(df_new)"""
        ax = fig.add_subplot (1 , 6 , i+1)
        plt.bar (datasets , means_fact
                 , color="black",alpha=0.4 )
        plt.bar (datasets , means_opinion
                 , color="red",alpha=0.4)

        print( "fact", dirs[i], sum(means_fact) /len (means_fact) )
        print ("opinion" , dirs[i] , sum (means_opinion) / len (means_opinion))
        #tips = sns.load_dataset(df["Inc"])
    #ax = sns.boxplot(y=df2['Inc'],palette="Set2")
    #tips = sns.load_dataset ("tips")
    #sns.distplot (df_list)
    #g = sns.FacetGrid (tips , col="sex" , hue="smoker")
    #bins = np.linspace (0 , 60 , 13)
    #g.map (plt.hist , "total_bill" , color="steelblue" , bins=bins)
    plt.subplots_adjust (hspace=0.3,wspace = 0.3)
    plt.show()
    fig.savefig (args.out_path)

if __name__ == '__main__':
    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')


    parser.add_argument ('--log_path_fact' , type=str , default="./data/dataset.csv" ,
                         help='input path of logs')

    parser.add_argument ('--log_path_opinion' , type=str , default='./data/sentiment_dataset_sentic.csv' ,
                         help='output of plots')

    args = parser.parse_args ()
    polar_fact_inc(args)

#gg(gg.mtcars, gg.aes( y=df["Inc"].values)) + gg.geom_boxplot()