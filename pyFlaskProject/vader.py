from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from os.path import isfile , join
import operator

import argparse


#---------------------// sentiment analysis with vader //------------------------------------------#

#call sentiment analysizer

analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    # score = analyzer.polarity_scores(sentence)
    # score.pop("compound")
    # return score
    neg_threshold = -0.05
    pos_threshold = 0.05
    score = analyser.polarity_scores(sentence)
    score['Compound'] = score.pop("compound")
    if score['Compound'] <= neg_threshold:
        score['Polarity'] = 'Negative'
    elif score['Compound'] > neg_threshold and score['Compound'] < pos_threshold:
        score['Polarity'] = 'Neutral'
    elif score['Compound'] >= pos_threshold:
        score['Polarity'] = 'Positive'
    return score['Polarity']

#MODEL ARGS
if __name__ == '__main__':

    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--input_path' , type=str , default="data/dataset.csv" ,
                         help='parse load path')
    parser.add_argument ('--out_path' , type=str , default='data/sentiment_dataset_vader.csv' ,
                         help='data save path')

    args = parser.parse_args()

    #df input
    df_input=pd.read_csv(args.input_path,sep=";", encoding = "ISO-8859-1")

    df_input.columns=["Id","Review","Golden"]

    #df output
    df_output= pd.DataFrame (columns=["Id","Review","Golden","Pred_sentiment"])

    for i in range (len (df_input)):

        sentimentData_sentence1 = sentiment_analyzer_scores (df_input.iloc[i]["Review"] )

        df_output.loc[i] = [df_input.iloc[i]["Id"] , df_input.iloc[i]["Review"] , df_input.iloc[i]["Golden"] ,  max(sentimentData_sentence1.items(), key=operator.itemgetter(1))[0]]

    df_output.to_csv (args.out_path, sep=";", index= False)
