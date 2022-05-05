import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
#Get files from exp directory
cmap = sns.diverging_palette (220 , 10 , as_cmap=True)
current_palette_7 = sns.color_palette("hls", 7)
sns.set_palette("Set2")

import numpy as np


def get_values(df):
    df1=df.query("Inc==1")
    #df2=df.query("Inc==0")
    index=df1.index

    X=[]
    Y=[]
    #s= df1["ID"].sum()
    for i in range(len(index)):
        x1=index[i][0]

        x2=index[i][1]
        df2=df.query("ration=="+str(x1))


        s = df2["Id"].sum ()

        y=(df1.loc[x1,x2]["Id"])


        X.append(x1)
        Y.append((y/s))

    return X,Y
def wmd_sim_density(args):
    #sns.set (style="whitegrid" , context="talk" , font_scale=0.6 , color_codes=True , palette="Set2")

    mypath= args.log_path
    mypath2 = args.log_path2


    dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
    dirs2 = [dI for dI in listdir (mypath2) if isdir (join (mypath2 , dI))]
    datasets=["AMAZON_PRODUCTS","NEWS_HEADLINES","STANFORD_TREEBANK"]
    fig = plt.figure ()
    #sns.set (style="whitegrid")
    #sns.set (font_scale=0.6)

    #sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
    nbdir=0

    for i in range(len(dirs)):
        path_dir= join(mypath,dirs[i])
        path_dir2=join(mypath2,dirs2[i])

        onlyfiles = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
        onlyfiles2 = [f for f in listdir (path_dir2) if isfile (join (path_dir2 , f))]
        nbdir=nbdir+1
        numfile=nbdir
        for k  in range (len(onlyfiles)):
            path_file=join(path_dir,onlyfiles[k])
            path_file2 = join (path_dir2 , onlyfiles2[k])

            df_wmd= pd.read_csv(path_file,sep=";")
            df_cos = pd.read_csv (path_file2 , sep=";")

            df1 = df_wmd.query ("Inc==1")
            df2 = df_cos.query ("Inc==1")
            print (df2.columns)
            #print(df.columns)

            ax = fig.add_subplot (3 , 6 , numfile)
            print(df1["Dist_WMD"])
            print(df2["Dist_euc"])
            sns.distplot (df1["Dist_WMD"].apply(lambda x: 10/(1+x)),ax=ax)
            sns.distplot (df2["Dist_euc"] .apply(lambda x: x*10), ax=ax )
            ax.set_xticks ([0,1,2,3,4,5,6 , 7,8,9,10 ])
            #ax.set_yticks ([0.6 , 0.7 , 0.8 , 0.9 , 1])
            #hist , bins = np.histogram (df["Inc"])
            #ax.bar (bins[:-1], hist.astype (np.float32) / hist.sum () , width=(bins[1] - bins[0]) , color='black')
            #plt.hist(df["Inc"],density=True)

            #ax.set (xlabel='X= Inconsistency degree' , ylabel='Y= % clusters whith inconsistency X')
            ax.set_ylabel("Y=  inconsistencies %", fontsize=8)
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


def sim_inc(args):
    #sns.set (style="whitegrid" , context="talk" , font_scale=0.6 , color_codes=True , palette="Set2")
    #markers = ["^" , "o" , "s"]
    shape = ["-" , "--" , "-."]

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
        h=0
        for f in onlyfiles:
            print(f)
            path_file = join (path_dir , f)
            df = pd.read_csv (path_file , sep=";")
            # df = df.loc[df["Dist_WMD"] <= 0.7]
            df= df.loc[df['Cos_sim'] != 1]
            df = df.loc[df['Dist'] != 0]
            df["ration"] = df["Cos_sim"] / df["Dist"].apply (lambda x: 1 / (1 + x))
            df["ration"] = df["ration"].apply (lambda x: round (x , 1))
            # print ("corr ",f ," ", df[["Dist_WMD" , "Inc"]].corr ())
            clusters = df.groupby (["ration" , "Inc"]).agg ("count")

            X , Y = get_values (clusters)
            # print ("corr coef" , f , "    " , pearsonr (X , Y))

            # sns.scatterplot (X , Y ,ax=ax)            #clusters= list(clusters)
            # y1=df["Inc"].apply(lambda x: round(x,2))
            ax = fig.add_subplot (1 , 6 , nbdir)
            sns.lineplot (X , Y , ax=ax,  markersize=6)
            ax.set_xticks ([0,0.2,0.4,0.6 , 0.8 , 1,1.2,1.4])
            ax.lines[h].set_linestyle (shape[h])
            h=h+1
            # hist , bins = np.histogram (df["Inc"])
            # ax.bar (bins[:-1], hist.astype (np.float32) / hist.sum () , width=(bins[1] - bins[0]) , color='black')
            # plt.hist(df["Inc"],density=True)
            # ax.set (xlabel='X= Inconsistency degree' , ylabel='Y= % clusters whith inconsistency X')
            ax.set_ylabel ("Y=inconsistencies %" , fontsize=6)
            ax.set_xlabel ('X= Cos Sim/wmd_sim' , fontsize=6)
            ax.set_title (dirs[nbdir-1] , fontsize=8)
            plt.xticks(fontsize=6)
            plt.yticks (fontsize=6)
            """
            if numfile <= 6:
                ax.set_title (dirs[numfile - 1] , fontsize=10)
            if numfile % 6 == 1:
                ax.set_ylabel (datasets[numfile // 6] , fontsize=10)
                ax.yaxis.set_label_position ("left")
            numfile = numfile + 6"""
            # ax.tick_params (axis='both' , which='major' , pad=5)
            # ax.set_xlabel('X= Inconsistency degree',fontsize=10)

            # ax.yaxis.set_label_position ("right")
    plt.figlegend ( datasets , loc='upper center' , ncol=3 , labelspacing=0.)
    plt.subplots_adjust (hspace=0.3 , wspace=0.3)
    plt.show ()
    fig.savefig (args.out_path)

if __name__ == '__main__':
    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--log_path' , type=str , default="./data/dataset.csv" ,
                         help='input path of logs')
    parser.add_argument ('--log_path2' , type=str , default="./data/dataset.csv" ,
                         help='input path of logs')

    parser.add_argument ('--out_path' , type=str , default='./data/sentiment_dataset_sentic.csv' ,
                         help='output of plots')

    args = parser.parse_args ()
    #wmd_sim_density(args)
    sim_inc(args)

#gg(gg.mtcars, gg.aes( y=df["Inc"].values)) + gg.geom_boxplot()