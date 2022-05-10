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