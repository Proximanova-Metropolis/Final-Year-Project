import argparse
import numpy as np
import gensim
import pandas as pd
from sentic import SenticPhrase
from os.path import isfile , join
import operator


#---------------------// sentiment analysis with sentic //------------------------------------------#
def sentiment_analyzer_scores(sentence):

    sp1 = SenticPhrase(sentence)

    score = sp1.get_sentiment (sentence)

    if "positive" in score: score = 'Positive'

    if "negative" in score: score = 'Negative'

    if "neutral" in score: score = 'Neutral'

    return score

if __name__ == '__main__':

    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--input_path', type=str , help='parse load path')

    parser.add_argument ('--out_path', type=str, help='data save path')

    args = parser.parse_args()

    #input dataframes
    df_input=pd.read_excel(args.input_path)

    #output dataframes
    df_output=pd.DataFrame (columns=["ID", "Comments", "Score"])

    df_input_comments = df_input[df_input.columns[:2]]

    sentimentData_sentence = []

    for i in range (len(df_input_comments)):

        sentimentData_sentence.append(sentiment_analyzer_scores(df_input_comments.iloc[i][1]))

        df_output["ID"] = df_input_comments["No."]
        df_output["Comments"] = df_input_comments["Comments"]
        df_output["Score"] = np.array(sentimentData_sentence)

    df_output.to_excel (args.out_path)