from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from os.path import isfile , join
import operator

import argparse


#---------------------// sentiment analysis with vader //------------------------------------------#

#call sentiment analysizer

analyzer = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    score.pop("compound")
    return score
