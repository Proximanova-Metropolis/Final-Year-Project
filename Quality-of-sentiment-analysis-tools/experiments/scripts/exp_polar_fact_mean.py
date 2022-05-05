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

cmap = sns.diverging_palette (220 , 10 , as_cmap=True)


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)
    plt.show()
    return axe

def polar_fact_inc(args):
    path_fact= args.log_path_fact
    path_opinion= args.log_path_opinion
    dirs_fact = [dI for dI in listdir (path_fact) if isdir (join (path_fact , dI))]
    dirs_opinion= dirs = [dI for dI in listdir (path_opinion) if isdir (join (path_opinion , dI))]
    datasets=["AMAZON\nREVIEWS","NEWS\nHEAD","STS"]
    tools = ["CHAR_CNN" , "CNN_TEXT" , "SENTICNET" , "SENTIWORDNET" , "STANFORD" , "VADER"]
    dfs=[]
    fig = plt.figure ()
    for i in range(len(dirs)):
        path_dirs_fact= join(path_fact,dirs_fact[i])
        path_dirs_opinion = join (path_opinion,dirs_opinion[i])

        onlyfiles_fact = [f for f in listdir(path_dirs_fact) if isfile(join(path_dirs_fact, f))]
        onlyfiles_opinion = [f for f in listdir(path_dirs_opinion) if isfile(join(path_dirs_opinion, f))]
        dfi=pd.DataFrame(columns=["polar_fact","subjective"], index=datasets)
        for j in range (len (onlyfiles_fact)):
            path_file_fact=join(path_dirs_fact,onlyfiles_fact[j])
            path_file_opinion = join (path_dirs_opinion , onlyfiles_opinion[j])
            df_fact= pd.read_csv(path_file_fact,sep=";")
            df_opinion = pd.read_csv (path_file_opinion , sep=";")
            dfi["subjective"].iloc[j]= (df_opinion["Inc"].mean ())
            dfi["polar_fact"].iloc[j] = (df_fact["Inc"].mean ())
        ax = fig.add_subplot (1 , 6 , i + 1)
        ax.set_title(tools[i])

        bars=dfi.plot(kind='bar',ax=ax,  width=0.9,  colors=["#00CED1","#ff6e6e"],linewidth=1.5,edgecolor="black", alpha=0.8)
        ax.get_legend ().remove ()
        ax.set_ylabel ("Mean inconsistency rate" , fontsize=10)

        #dfs.append(dfi)
    #plot_clustered_stacked (dfs , tools)

    #ax = fig.add_subplot (1 , 6 , i+1)
    #plt.hist (  [means_fact,means_opinion],datasets,stacked=True , color="black",alpha=0.4 )
   # plt.bar (datasets , means_opinion, color="red",alpha=0.4)

    #print( "fact", dirs[i], sum(means_fact) /len (means_fact) )
    #print ("opinion" , dirs[i] , sum (means_opinion) / len (means_opinion))
    #plt.subplots_adjust (hspace=0.3,wspace = 0.3)

    plt.figlegend (["polar_fact","subjective"] , loc='right' , ncol=2 , labelspacing=0.)
    plt.show()
    #fig.savefig (args.out_path)

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