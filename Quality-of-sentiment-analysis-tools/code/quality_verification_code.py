# Author: Wissam  Mammar kouadri

from nltk.corpus import stopwords
import os
from gensim.models import Word2Vec
import gensim
from nltk import word_tokenize
from nltk import download
from gensim.similarities import WmdSimilarity
import json
import argparse
import pandas as pd 
#download('stopwords')


def calculate_thershold(input_file, out_file,w2v_path):
        #Load datasets
        pd.options.mode.chained_assignment = None
        df= pd.read_csv(input_file, sep=";")
        #get stop words
        stop_words = stopwords.words('english')

        #load word2vec model

        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        #add new column

        df['D_wmd']= 100
        #preprocess
        for i in range (len(df)):
            #normalize lower case
            df.at[i,"S1"]= df.loc[i]["S1"].lower().split()
            df.at[i,"S2"]= df.loc[i]["S2"].lower().split()

            ##delete stop words
            df.at[i,"S1"]=[w for w in df.loc[i]["S1"] if w not in stop_words]
            df.at[i,"S2"]=[w for w in df.loc[i]["S2"] if w not in stop_words]

            #calculate distance betweeb s1 and s2
            df.at[i,"D_wmd"]= model.wmdistance(df.loc[i]["S1"], df.loc[i]["S2"])



        median_similarity = df["D_wmd"].median()
        df.to_csv(out_file, sep=";")
        return median_similarity

		
def clean_dataset(input_file, out_file, t, w2v_path):
        # download('stopwords')
        # Load datasets
        pd.options.mode.chained_assignment = None
        df_old = pd.read_csv (input_file , sep=";")
        df_new = pd.DataFrame (columns=["Id" , "Review" , "Golden" , "Pred" , "Dist"])


        # load word2vec model

        model = gensim.models.KeyedVectors.load_word2vec_format (w2v_path , binary=True)

        # groupe by ID
        list_groups = (list (df_old.groupby ("Id")))

        for group in list_groups:
            cluster = (group[1]).reset_index (drop=True)
            if len (cluster) >= 2:
                golden_sentence = cluster.loc[0]["Review"]
                df_new.loc[len (df_new)] = cluster.loc[0]
                for i in range (len (cluster)):
                    dist = model.wmdistance (golden_sentence , cluster.loc[i]["Review"])
                    if dist < t:
                        df_new.loc[len (df_new)] = [cluster.iloc[i]['Id'] , cluster.iloc[i]['Review'] ,
                                                    cluster.iloc[i]['Golden'] , cluster.iloc[i]['Pred'] , dist]



        df_new=df_new.drop_duplicates(subset='Review')
        df_new.to_csv (out_file , sep=";" , index=False)

def main(args):
    input_file_paraphrase = args.input_file_paraphrase
    out_file_paraphrase=args.out_file_paraphrase
    w2v_path= args.w2v_path
    input_file_sentiment= args.input_file_sentiment
    out_file_sentiment = args.out_file_sentiment
    t= calculate_thershold(input_file_paraphrase,out_file_paraphrase,w2v_path)
    clean_dataset(input_file_sentiment,out_file_sentiment,t,w2v_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='verification of data quality of the generated dataset')
    parser.add_argument( '--input_file_paraphrase' , type=str , default='../data/paraphrases_dataset/sts.csv' ,
                         help='path to paraphrases dataset')
						 
    parser.add_argument( '--out_file_paraphrase' , type=str , default='../data/paraphrases_dataset/sts.csv' ,
                         help='path to save paraphrases dataset with distance')
    parser.add_argument ('--input_file_sentiment' , type=str , default='../data/sentiment_dataset/news.csv' ,
                         help='path to sentiment dataset')
    parser.add_argument ('--out_file_sentiment' , type=str , default='../data/clea_datasets/news.csv' ,
                         help='path to sentiment dataset')
    parser.add_argument('--w2v_path' , type=str , default='../data/models/GoogleNews-vectors-negative300.bin' ,
                         help='path to w2v embedding file')
    args = parser.parse_args ()
    main(args)









