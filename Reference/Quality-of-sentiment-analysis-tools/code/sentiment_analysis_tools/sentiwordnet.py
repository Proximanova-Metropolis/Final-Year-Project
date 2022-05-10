import nltk
#nltk.download()


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize , word_tokenize , pos_tag
import pandas as pd
from os.path import isfile , join
import operator
import argparse
lemmatizer = WordNetLemmatizer ()


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith ('J'):
        return wn.ADJ
    elif tag.startswith ('N'):
        return wn.NOUN
    elif tag.startswith ('R'):
        return wn.ADV
    elif tag.startswith ('V'):
        return wn.VERB
    return None


def clean_text(text):
    text = text.replace ("<br />" , " ")
    return text


def sentiment_analyzer_scores(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """

    sentiment = 0.0
    tokens_count = 0

    text = clean_text (text)

    raw_sentences = sent_tokenize (text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag (word_tokenize (raw_sentence))

        for word , tag in tagged_sentence:
            wn_tag = penn_to_wn (tag)
            if wn_tag not in (wn.NOUN , wn.ADJ , wn.ADV):
                continue

            lemma = lemmatizer.lemmatize (word , pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets (lemma , pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset (synset.name ())

            sentiment += swn_synset.pos_score () - swn_synset.neg_score ()
            tokens_count += 1

    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0

    # sum greater than 0 => positive sentiment
    if sentiment > 0:
        return "Positive"

    # negative sentiment
    if sentiment <0 :
        return "Negative"

    if sentiment == 0:
        return "Neutral"




if __name__ == '__main__':

    #model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--input_path' , type=str , default="data/dataset.csv" ,
                         help='parse load path')
    parser.add_argument ('--out_path' , type=str , default='data/sentiment_dataset_sentiwn.csv' ,
                         help='data save path')

    args = parser.parse_args()

    #iput df
    df_input=pd.read_csv(args.input_path,sep=";", encoding = "ISO-8859-1", on_bad_lines='skip', lineterminator='\n')

    #df_input.columns=["Id","Review","Golden"]

    #output df
    df_output= pd.DataFrame (columns=["Id","Review","Golden","Pred"])


    for i in range (len (df_input)):

        sentimentData_sentence1 = sentiment_analyzer_scores (df_input.iloc[i]["Review"])

        df_output.loc[i] = [df_input.iloc[i]["Id"] , df_input.iloc[i]["Review"] ,
                            df_input.iloc[i]["Golden"],sentimentData_sentence1 ]

    df_output.to_csv (args.out_path, sep=";", index=False)



    """
    sentimentData_sentence1 = sentiment_analyzer_scores ("investment")
    print(sentimentData_sentence1)"""