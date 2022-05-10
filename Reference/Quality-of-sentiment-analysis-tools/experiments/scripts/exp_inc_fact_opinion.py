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
#current_palette_7 = sns.color_palette("hls", 7)
#sns.set_palette(current_palette_7)
from sklearn.metrics import accuracy_score


def get_inc_mean_fact(input_path,out_path):
    dirs= [dI for dI in listdir (input_path) if isdir (join (input_path , dI))]
    datasets=["AMAZON_PRODUCTS","NEWS_HEADLINES","STANFORD_TREEBANK"]
    r=[1,2,3]
    fig = plt.figure ()

    for i in range (len (dirs)):
        path_dirs_fact = join (input_path , dirs[i])
        onlyfiles = [f for f in listdir (path_dirs_fact) if isfile (join (path_dirs_fact , f))]
        means_fact_fact = 0
        means_fact_opinion = 0
        means_opinion_opinion = 0
        meanopinion_fact1 = 0
        meanopinion_fact2 = 0
        dfi=pd.DataFrame(columns=["fact_fact","sub_sub","sub_fact"],index=datasets)
        facts=[]
        subjectives=[]
        for j in range (len (onlyfiles)):
            path_file_fact = join (path_dirs_fact , onlyfiles[j])

            df = pd.read_csv (path_file_fact , sep=";")

            clusters1 = df.groupby (["Nature1"])

            clusters = df.groupby (["Nature1" , "Nature2" , "Inc"]).agg ('count')
            # clusters= list(clusters)

            print("---------------------")
            clusters1=(list(clusters1))


            fact= clusters1[0][1]
            subjective = clusters1[1][1]
            subjectives.append( accuracy_score(subjective["Golden"],subjective["Pred1"]))


            facts.append ( accuracy_score(fact["Golden"],fact["Pred1"]))

            res = [list (ele) for ele in clusters.index.values]

            if ["Fact" , "Fact" , 1] in res:
                means_fact_fact = (clusters.loc["Fact" , "Fact" , 1]["s1"] / (
                            clusters.loc["Fact" , "Fact" , 1]["s1"] + clusters.loc["Fact" , "Fact" , 0]["s1"]))

            if ["Subjective" , "Subjective" , 1] in res:
                means_opinion_opinion = (clusters.loc["Subjective" , "Subjective" , 1]["s1"] / (
                            clusters.loc["Subjective" , "Subjective" , 1]["s1"] +
                            clusters.loc["Subjective" , "Subjective" , 0]["s1"]))

            if ["Subjective" , "Fact" , 1] in res and ["Subjective" , "Fact" , 0] in res:
                meanopinion_fact1 = clusters.loc["Subjective" , "Fact" , 1]["s1"] / (
                            clusters.loc["Subjective" , "Fact" , 1]["s1"] + clusters.loc["Subjective" , "Fact" , 0][
                        "s1"])

            else:
                if ["Subjective" , "Fact" , 1] in res and ["Subjective" , "Fact" , 0] not in res:
                    meanopinion_fact1 = 1

            if ["Fact" , "Subjective" , 1] in res and ["Fact" , "Subjective" , 0] in res:
                meanopinion_fact2 = clusters.loc["Fact" , "Subjective" , 1]["s1"] / (
                            clusters.loc["Fact" , "Subjective" , 1]["s1"] + clusters.loc["Fact" , "Subjective" , 0][
                        "s1"])

            else:
                if ["Fact" , "Subjective" , 1] in res: meanopinion_fact2 = 1

            means_fact_opinion = ((meanopinion_fact1 + meanopinion_fact2) / 2)
            #print (means_fact_opinion , means_opinion_opinion , means_fact_fact)

            means_fact_opinion= ((meanopinion_fact1 + meanopinion_fact2)/2)
            dfi["fact_fact"].iloc[j]= means_fact_fact
            dfi["sub_sub"].iloc[j]  = means_opinion_opinion
            dfi["sub_fact"].iloc[j] = means_fact_opinion

        print("fact_acc", facts)
        print ("fact_acc_mean" , sum(facts)/len(facts))
        print ("subjective_acc" , subjectives)
        print ("subjectives_acc_mean" , sum (subjectives) / len (subjectives))
        ax = fig.add_subplot (1 , 6, i + 1)
        plt.xticks (fontsize=8)
        plt.yticks (rotation=45 , fontsize=8)
        ax.set_title (dirs[i])
        # From raw value to percentage
        print(dirs[i])
        #print(dfi ,"dfi")
        #print(dfi["fact_fact"].mean(), "fact_fact")
        #print (dfi["sub_sub"].mean () , "sub_sub")
        #print (dfi["sub_fact"].mean () , "sub_fact")

        totals = [h + z + k for h , z , k in zip (dfi['fact_fact'] , dfi['sub_sub'] , dfi['sub_fact'])]
        #totals= [(lambda x: 1 if x==0 else x)(x) for x in totals]
        greenBars = [h / z * 100 for h , z in zip (dfi['fact_fact'] , totals)]
        orangeBars = [h / z * 100 for h , z in zip (dfi['sub_sub'] , totals)]
        blueBars = [h / z * 100 for h , z in zip (dfi['sub_fact'] , totals)]
        """
        greenBars=dfi["fact_fact"]
        orangeBars=dfi["sub_sub"]
        blueBars= dfi["sub_fact"]"""
        # plot
        barWidth = 0.6
        names = ("AMAZON\nREVIEWS","NEWS\nHEAD","STS")
        # Create green Bars
        plt.bar (r , greenBars , color='#FFC300' , edgecolor='white' , width=barWidth)
        # Create orange Bars
        plt.bar (r , orangeBars , bottom=greenBars , color='#FF5733' , edgecolor='white' , width=barWidth)
        # Create blue Bars
        plt.bar (r , blueBars , bottom=[k + z for k , z in zip (greenBars , orangeBars)] , color='#C70039' ,
                 edgecolor='white' , width=barWidth)
        plt.xticks (r , names)
        ax.set_ylabel ("% of inconsistencies")

        # Custom x axis


        #bars = dfi.plot (kind='bar' , ax=ax , width=0.9 , linewidth=1.5 , edgecolor="black" , alpha=0.8)
        #ax.get_legend ().remove ()
        #ax.set_ylabel ("Mean inconsistency rate" , fontsize=10)
    plt.figlegend (["fact_fact" , "sub_sub","sub_fact"] , loc='upper center' , ncol=3 , labelspacing=0.)



    plt.show ()
if __name__ == '__main__':
    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')


    parser.add_argument ('--input_path' , type=str , default="D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/public_folder/experiments/logs/intool_inc_polar_fact" ,
                         help='input path of logs')

    parser.add_argument ('--out_path' , type=str , default='./data/sentiment_dataset_sentic.csv' ,help='output of plots')

    args = parser.parse_args ()
    get_inc_mean_fact(args.input_path, args.out_path)
