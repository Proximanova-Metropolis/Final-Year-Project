import pandas as pd

import argparse
from os import listdir

from os.path import isfile, join

def quality(input_path, out_path):

    onlyfiles = [join( input_path,f) for f in listdir (input_path) if isfile (join (input_path , f))]
    datasets = [f for f in listdir (input_path) if isfile (join (input_path , f))]

    df_new=pd.DataFrame(columns=["Dataset","Negative", "Positive","Neutral","Culuster","Total"])
    total_clusters= 0
    total_pos=0
    total_neg=0
    total_neut=0
    total_all=0
    i=0
    for file in onlyfiles:
        print(file)
        df= pd.read_csv(file, sep=";")
        total_all=len(df)+total_all

        cluster_pol= df.groupby("Golden").count()
        clusters= df.groupby("Id")
        clusters=list(clusters)
        total_clusters=total_clusters+len(clusters)
        print(cluster_pol)

        total_neg=total_neg+ cluster_pol['Id'].loc["Negative"]
        total_pos = total_pos + cluster_pol['Id'].loc["Positive"]
        total_neut = total_neut + cluster_pol['Id'].loc["Neutral"]
        df_new.loc[i]=[datasets[i],cluster_pol["Id"].loc["Negative"],cluster_pol['Id'].loc["Positive"] ,cluster_pol['Id'].loc["Neutral"], len(clusters), len(df)]
        i=i+1
    df_new.loc[i] = ["Total_data" , total_neg ,total_pos ,
                     total_neut , total_clusters , total_all]


    df_new.to_csv(out_path, sep=";", index=False)
    print(df_new)
if __name__ == '__main__':
    # Ignore warning message by tensor flow

    # model args
    parser = argparse.ArgumentParser (description='Normalize datasets')

    parser.add_argument ('--input_path' , type=str , default="../data/sentiment_datasets_augmented"
                                                             ,
                                                              help=' load path')
    parser.add_argument ('--out_path' , type=str , default='../data/sentiment_datasets_labels' ,
                         help='data save path')

    args = parser.parse_args()
    quality(args.input_path, args.out_path)
