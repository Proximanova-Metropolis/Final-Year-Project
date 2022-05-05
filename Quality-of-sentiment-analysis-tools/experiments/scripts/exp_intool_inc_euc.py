import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
from scipy.stats.stats import pearsonr
#Get files from exp directory

sns.set(style="whitegrid", context="talk",font_scale=0.6,color_codes=True,palette ="Set2")
#color_palette ("RdBu" , n_colors=7)

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
    #get values of inconsistency
    df1=df.query("Inc==1")

    index=df1.index

    X=[]
    Y=[]
    #s= df1["ID"].sum()
    for i in range(len(index)):
        #get the cos similarity
        x1=index[i][0]


        x2=index[i][1]

        #get pairs with distance x1
        df2=df.query("Dist_euc=="+str(x1))
        #print ("df2 " , df2)

        #sum pairs with distance x1
        s = df2["ID"].sum ()
        #print ("s " , s)
        #get count of inconsistente pairs with distance x1
        y=(df1.loc[x1,x2]["ID"])




        #add x1 to X axis
        X.append(x1)
        #add proportion of inconsistente pairs to Y axis
        Y.append(y/s)

    return X,Y



def sim_inc(args):
        mypath = args.log_path
        #get dirs (tools)
        dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
        datasets = ["AMAZON_PRODUCTS" , "NEWS_HEADLINES" , "STANFORD_TREEBANK"]
        tools = ["CHAR_CNN" , "CNN_TEXT" , "SENTICNET" , "SENTIWORDNET" , "STANFORD" , "VADER"]
        fig = plt.figure ()
        # sns.set (style="whitegrid")

        # sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
        numfile = 0
        for dir in dirs:
            path_dir = join (mypath , dir)
            #get files of each tool
            onlyfiles = [f for f in listdir (path_dir) if isfile (join (path_dir , f))]
            numfile = numfile + 1
            j=0

            markers=["^","o","s"]
            shape=["-","--",":"]
            h=0
            for f in onlyfiles:


                path_file=join(path_dir,f)
                #read the dataset
                df= pd.read_csv(path_file,sep=";")

                #round values
                df["Dist_euc"]= df["Dist_euc"].apply(lambda x: round(x,2))

                #group results by distance and INC
                clusters=df.groupby(["Dist_euc","Inc"]).agg("count")

                #get Y and X axis
                X,Y=get_values (clusters)
                print ("corr coef" , f , "    " ,pearsonr (X , Y) )






                #sns.scatterplot (X , Y ,ax=ax)            #clusters= list(clusters)
                #y1=df["Inc"].apply(lambda x: round(x,2))



                ax = fig.add_subplot (1, 6 , numfile)
                sns.lineplot (X,Y , ax=ax)
                ax.lines[h].set_linestyle (shape[h])
                ax.set_xticks ([0,0.2,0.4,0.6,0.8,1])
                h=h+1
                j=j+1
                #print(X)
                #print(y1)
            ax.set_ylabel ("Inconsistency rate for Cos_sim X" , fontsize=10)
            ax.set_xlabel ("Cos similarity" , fontsize=10)
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