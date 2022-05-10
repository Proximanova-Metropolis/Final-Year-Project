
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
import numpy as np
#Get files from exp directory
def intertool_inc(args):
    mypath = args.log_path
    # dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
    datasets = ["AMAZON_PRODUCTS" , "NEWS_HEADLINES" , "STANFORD_TREEBANK"]
    tools = ["char_cnn" , "rec_nn" , "senticnet","sentiwordnet"  , "text_cnn" , "vader"]
    fig = plt.figure ()
    # sns.set (style="whitegrid")

    # sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
    nbdir = 0

    # for dir in dirs:
    # path_dir = join (mypath , dir)
    onlyfiles = [f for f in listdir (mypath) if isfile (join (mypath , f))]
    # nbdir = nbdir + 1
    numfile = 0
    for f in onlyfiles:
        path_file = join (mypath , f)
        df = pd.read_csv (path_file , sep=";")
        df.columns = ["Id" , "char_cnn" , "rec_nn" , "senticnet","sentiwordnet"  , "text_cnn" , "vader"]
        # print(df.columns)
        for column in tools:
            numfile = numfile + 1
            ax = fig.add_subplot (3 , 6 , numfile)

            #sns.distplot (df[column] , ax=ax , color="black")
            hist , bins = np.histogram (df[column])
            ax.bar (bins[:-1] , hist.astype (np.float32) / hist.sum () , width=(bins[1] - bins[0]),color="#088A85", alpha=0.5)
            #ax.set (xlabel='X=Inc degree' , ylabel='Rate A with inc X',fontsize=7)
            plt.xticks ([0 , 0.2 , 0.4 , 0.6 , 0.8 , 1], fontsize=6)
            plt.yticks ([0 , 0.1 , 0.15 , 0.2 , 0.25 ,0.3,0.4,0.5],fontsize=6)
            ax.set_ylabel ("Rate of A with inc X" , fontsize=7)
            ax.set_xlabel ('X= Inc degree' , fontsize=7)
            if numfile <= 6:
                ax.set_title (tools[numfile - 1] , fontsize=7)

            if numfile % 6 == 1:
                ax.set_ylabel (datasets[numfile // 6] , fontsize=7)
                ax.yaxis.set_label_position ("left")

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
    intertool_inc(args)

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """
"""Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""
"""
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

# create fake dataframes
df1 = pd.DataFrame(np.random.rand(4, 5),
                   index=["A", "B", "C", "D"],
                   columns=["I", "J", "K", "L", "M"])
df2 = pd.DataFrame(np.random.rand(4, 5),
                   index=["A", "B", "C", "D"],
                   columns=["I", "J", "K", "L", "M"])
df3 = pd.DataFrame(np.random.rand(4, 5),
                   index=["A", "B", "C", "D"],
                   columns=["I", "J", "K", "L", "M"])

# Then, just call :
plot_clustered_stacked([df1, df2, df3],["df1", "df2", "df3"])
"""