import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
#Get files from exp directory


import numpy as np



def wmd_sim_density(args):
    sns.set (style="whitegrid" , context="talk" , font_scale=0.6 , color_codes=True , palette="Set2")

    mypath= args.log_path
    dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
    datasets=["AMAZON_PRODUCTS","NEWS_HEADLINES","STANFORD_TREEBANK"]
    fig = plt.figure ()
    #sns.set (style="whitegrid")
    #sns.set (font_scale=0.6)

    #sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
    nbdir=0

    for dir in dirs:
        path_dir= join(mypath,dir)
        onlyfiles = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
        nbdir=nbdir+1
        numfile=nbdir
        for f in onlyfiles:
            path_file=join(path_dir,f)
            df= pd.read_csv(path_file,sep=";")

            df1 = df.query ("Inc==1")
            #print(df.columns)

            ax = fig.add_subplot (3 , 6 , numfile)
            sns.distplot (df1["Dist_WMD"].apply(lambda x: 10/(1+x)),ax=ax, bins=50)
            ax.set_xticks ([6 , 7,8,9,10 ])
            #ax.set_yticks ([0.6 , 0.7 , 0.8 , 0.9 , 1])
            #hist , bins = np.histogram (df["Inc"])
            #ax.bar (bins[:-1], hist.astype (np.float32) / hist.sum () , width=(bins[1] - bins[0]) , color='black')
            #plt.hist(df["Inc"],density=True)

            #ax.set (xlabel='X= Inconsistency degree' , ylabel='Y= % clusters whith inconsistency X')
            ax.set_ylabel("Y= number of inconsistencies", fontsize=8)
            ax.set_xlabel ('X= WMD Sim x 10^-1' , fontsize=8)
            ax.grid (False)

            if numfile<=6 :
                ax.set_title (dirs[numfile-1],fontsize=10)

            if numfile%6 ==1:


                ax.set_ylabel (datasets[numfile//6],fontsize=10)
                ax.yaxis.set_label_position ("left")

            numfile = numfile + 6
            #ax.tick_params (axis='both' , which='major' , pad=5)
            #ax.set_xlabel('X= Inconsistency degree',fontsize=10)


            #ax.yaxis.set_label_position ("right")


    plt.subplots_adjust (hspace=0.3,wspace = 0.3)
    plt.show()
    fig.savefig (args.out_path)

def cos_sim_density(args):
        sns.set (style="whitegrid" , context="talk" , font_scale=0.6 , color_codes=True , palette="Set2")

        mypath = args.log_path
        dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
        datasets = ["AMAZON_PRODUCTS" , "NEWS_HEADLINES" , "STANFORD_TREEBANK"]
        fig = plt.figure ()
        # sns.set (style="whitegrid")
        # sns.set (font_scale=0.6)

        # sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
        nbdir = 0

        for dir in dirs:
            path_dir = join (mypath , dir)
            onlyfiles = [f for f in listdir (path_dir) if isfile (join (path_dir , f))]
            nbdir = nbdir + 1
            numfile = nbdir
            for f in onlyfiles:
                path_file = join (path_dir , f)
                df = pd.read_csv (path_file , sep=";")

                df1 = df.query ("Inc==1")

                # print(df.columns)

                ax = fig.add_subplot (3 , 6 , numfile)
                sns.distplot (df1["Dist_euc"].apply (lambda x: 10 * x) , ax=ax , bins=50 , kde=True)
                ax.set_xticks ([0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10])
                # ax.set_yticks ([0.6 , 0.7 , 0.8 , 0.9 , 1])
                # hist , bins = np.histogram (df["Inc"])
                # ax.bar (bins[:-1], hist.astype (np.float32) / hist.sum () , width=(bins[1] - bins[0]) , color='black')
                # plt.hist(df["Inc"],density=True)

                # ax.set (xlabel='X= Inconsistency degree' , ylabel='Y= % clusters whith inconsistency X')
                ax.set_ylabel ("Y= number of inconsistencies" , fontsize=8)
                ax.set_xlabel ('X= Cos Sim x 10^-1' , fontsize=8)
                ax.grid (False)

                if numfile <= 6:
                    ax.set_title (dirs[numfile - 1] , fontsize=10)

                if numfile % 6 == 1:
                    ax.set_ylabel (datasets[numfile // 6] , fontsize=10)
                    ax.yaxis.set_label_position ("left")

                numfile = numfile + 6

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
        # tips = sns.load_dataset(df["Inc"])
        # ax = sns.boxplot(y=df2['Inc'],palette="Set2")
        # tips = sns.load_dataset ("tips")
        # sns.distplot (df_list)
        # g = sns.FacetGrid (tips , col="sex" , hue="smoker")
        # bins = np.linspace (0 , 60 , 13)
        # g.map (plt.hist , "total_bill" , color="steelblue" , bins=bins)
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
    wmd_sim_density(args)
    #cos_sim_density(args)

#gg(gg.mtcars, gg.aes( y=df["Inc"].values)) + gg.geom_boxplot()