import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
import math
from scipy.stats.stats import pearsonr

#Get files from exp directory
sns.set(style="whitegrid", context="talk",font_scale=0.6,color_codes=True,palette ="Set2")

"""
def get_values(df):
    df=df.query("Inc==1")
    index=df.index
    X=[]
    Y=[]
    s= df["Id"].sum()
    for i in range(len(index)):
        x1=index[i][0]
        x2=index[i][1]
        y=(df.loc[x1,x2]["Id"])
        if (x1<0.8):
         #y=df.query("Inc=="+ str(x2 )+"and Dist=="+ str(x1))

         X.append(x1)
         Y.append(y/s)
    print(sum(Y))
    return X,Y
"""
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
        df2=df.query("Dist=="+str(x1))


        s = df2["Id"].sum ()

        y=(df1.loc[x1,x2]["Id"])


        X.append(1/(1+x1))
        Y.append((y/s))

    return X,Y



def sim_inc(args):
    mypath = args.log_path
    dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
    datasets = ["AMAZON_PRODUCTS" , "NEWS_HEADLINES" , "STANFORD_TREEBANK"]
    tools = ["CHAR_CNN" , "REC_NN" , "SENTIWORDNET" , "SENTICNET" , "TXT_CNN" , "VADER"]
    fig = plt.figure ()
    # sns.set (style="whitegrid")
    #sns.set (font_scale=0.6)

    # sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
    numfile = 0
    for dir in dirs:
        path_dir = join (mypath , dir)
        onlyfiles = [f for f in listdir (path_dir) if isfile (join (path_dir , f))]
        numfile = numfile + 1
        j=0
        markers = ["^" , "o" , "s"]
        shape = ["-" , "--" , ":"]
        h = 0
        for f in onlyfiles:

            path_file=join(path_dir,f)
            df= pd.read_csv(path_file,sep=";")
            #df = df.loc[df["Dist_WMD"] <= 0.7]
            df["Dist"]= df["Dist"].apply(lambda x: round(x,1))

            #print ("corr ",f ," ", df[["Dist_WMD" , "Inc"]].corr ())
            clusters=df.groupby(["Dist","Inc"]).agg("count")
            X,Y=get_values (clusters)
            print ("corr coef" , f , "    " , pearsonr (X , Y))





            #sns.scatterplot (X , Y ,ax=ax)            #clusters= list(clusters)
            #y1=df["Inc"].apply(lambda x: round(x,2))



            ax = fig.add_subplot (1, 6 , numfile)
            sns.lineplot (X , Y , ax=ax )
            ax.lines[h].set_linestyle (shape[h])
            ax.set_xticks ([ 0.6,0.7 , 0.8 ,0.9, 1])
            h = h + 1
            j=j+1

            #print(X)
            #print(y1)

        ax.set_ylabel ("Inconsistency rate for WMD_Similarity X" , fontsize=10)
        ax.set_xlabel ("WMD Similarity (1/(1+WMD))" , fontsize=10)
        ax.set_title (dir , fontsize=10)

    plt.figlegend ( datasets , loc='lower center' , ncol=3 , labelspacing=0.)
    plt.show()
    #fig.savefig (args.out_path)

if __name__ == '__main__':
    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    #parser.add_argument ('--log_path' , type=str , default="D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/experiments/logs/Intool_inc_sim/sim_inc" ,help='input path of logs')
    parser.add_argument ('--log_path' , type=str , default="D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/experiments/logs/logs_dev/plot" ,help='input path of logs')



    parser.add_argument ('--out_path' , type=str , default='./data/sentiment_dataset_sentic.csv' ,
                         help='output of plots')

    args = parser.parse_args ()
    sim_inc(args)