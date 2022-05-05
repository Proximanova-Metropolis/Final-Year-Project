import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
import numpy as np
#Get files from exp directory



def intool_inc(args):
    #sns.set (style="white" , context="talk" , font_scale=0.8 , color_codes=True , palette="Set2")
    cmap = sns.diverging_palette (220 , 10 , as_cmap=True)

    mypath= args.log_path
    dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
    datasets = ["AMAZON_PRODUCTS" , "NEWS_HEADLINES" , "STANFORD_TREEBANK"]
    tools = ["CHAR_CNN" , "REC_NN"  , "SENTICNET" ,"SENTIWORDNET", "TXT_CNN" , "VADER"]

    fig = plt.figure ()
    #sns.set (style="whitegrid")
    #sns.set (font_scale=0.6)

    #sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
    numfile = 0
    for dir in dirs:
        path_dir= join(mypath,dir)
        onlyfiles = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]

        matrices=[]
        for f in onlyfiles:
            path_file=join(path_dir,f)
            df= pd.read_csv(path_file,sep=";")
            matrix = [
                [round (sum (df['p_p']) / len (df['p_p']) , 3) , round (sum (df['Inc_p_o']) / len (df['Inc_p_o']) , 3) ,
                 round (sum (df['Inc_p_n']) / len (df['Inc_p_n']) , 3)] ,
                [round (sum (df['Inc_p_o']) / len (df['Inc_p_o']) , 3) , round (sum (df['o_o']) / len (df['o_o']) , 3) ,
                 round (sum (df['Inc_o_n']) / len (df['Inc_o_n']) , 3)] ,
                [round (sum (df['Inc_p_n']) / len (df['Inc_p_n']) , 3) ,
                 round (sum (df['Inc_o_n']) / len (df['Inc_o_n']) , 3) , round (sum (df['n_n']) / len (df['n_n']) , 3)]]
            matrices.append (matrix)
        print(np.array (matrices[0]),"__________" , np.array (matrices[1]) ,"_________", np.array (matrices[2]))
        print("__________________________")
        summatrices =np.add( np.add (np.array (matrices[0]) , np.array (matrices[1])) , np.array (matrices[2]))/3
        numfile = numfile + 1


        ax = fig.add_subplot (1 , 6 , numfile)
        df_cm = pd.DataFrame (summatrices , ["pos","neut","neg"] ,  ["pos","neut","neg"])
        mask = np.zeros_like (df_cm , dtype=np.bool)
        mask[np.triu_indices_from (mask)] = True
        mask[np.diag_indices_from (mask)] = False
        if numfile !=6 :
            sns.heatmap (df_cm, annot=True, mask=mask ,cmap=cmap,linewidths=.5,ax=ax, cbar=False)# font size
        else:
            sns.heatmap (df_cm , annot=True , mask=mask , cmap=cmap , linewidths=.5 , ax=ax )  # font size
        ax.set_ylim (3 , 0)



        ax.set_title (dir,fontsize=8)


                #ax.yaxis.set_label_position ("left")



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
    #plt.subplots_adjust (hspace=0.2,wspace = 0.2)
    plt.show()
    #fig.savefig (args.out_path)

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