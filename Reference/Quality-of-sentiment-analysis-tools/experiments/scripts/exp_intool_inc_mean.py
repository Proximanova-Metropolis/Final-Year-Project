import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
#Get files from exp directory


current_palette_7 = sns.color_palette("hls", 7)
sns.set_palette(current_palette_7)
def intool_inc(args):
    mypath= args.log_path
    dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
    datasets=["AMAZONs","NEWS","STANFORD"]
    tools = ["CHAR_CNN" , "REC_NN" , "SENTIWORDNET" , "SENTICNET" , "TXT_CNN" , "VADER"]
    fig = plt.figure ()
    #sns.set (style="whitegrid")
    sns.set (font_scale=0.6)

    #sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
    nbdir=0

    for dir in dirs:
        path_dir= join(mypath,dir)
        onlyfiles = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
        nbdir=nbdir+1
        numfile=nbdir
        means=[]
        for f in onlyfiles:
            path_file=join(path_dir,f)
            df= pd.read_csv(path_file,sep=";")
            means.append( df["Inc"].mean())
            print (f, df["Inc"].mean())

            #print(df.columns)
        print(means)
        print (sum(means)/len(means))

        ax = fig.add_subplot (1 , 6 , nbdir)
        plt.bar (datasets , means, color="black")

        #sns.distplot (x=["Amazon","Stanford", "News"],Y=means , ax=ax , color="black")
        ax.set (xlabel='X= Inconsistency mean' , ylabel='Y= Nb clusters that have inconsistency X')
        #ax.grid (False)


        ax.set_title (dirs[nbdir-1],fontsize=10)








            #ax.tick_params (axis='both' , which='major' , pad=5)
            #ax.set_xlabel('X= Inconsistency degree',fontsize=10)


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

    parser.add_argument ('--log_path' , type=str , default="./data/dataset.csv" ,
                         help='input path of logs')

    parser.add_argument ('--out_path' , type=str , default='./data/sentiment_dataset_sentic.csv' ,
                         help='output of plots')

    args = parser.parse_args ()
    intool_inc(args)

#gg(gg.mtcars, gg.aes( y=df["Inc"].values)) + gg.geom_boxplot()