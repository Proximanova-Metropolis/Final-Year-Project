import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from gensim import models
import numpy as np
from torch import nn
import torch
import pickle
import os

PROJECT_PATH = os.getcwd ()


# load training dataset from a  give path and return a formating dataset ready to use
# @:param
#       path: th path of the dataset type: string
#       PATH_W2V: path too the word embbeding type: string
#       label_type: the label formating type normal for normal encoding and binary for onehot type: string
# @:returns
#        X: encoded reviews type numpy.array
#        Y: encoded labels type numpy.array


def load_dataset(path , PATH_W2V, label_type="normal"):
    # read the csv
    # df_dataset = pd.read_csv (path , sep=";" , encoding="utf-8")

    # read excel
    df_dataset = pd.read_excel(path, encoding="utf8", errors='ignore')

    # delete no ascii chars and tokenize
    df_dataset['Review'] = df_dataset["Review"].apply (
        lambda x: ''.join ([" " if ord (i) < 32 or ord (i) > 126 else i for i in x]))
    df_dataset['Review'] = df_dataset["Review"].apply (word_tokenize)

    # formating labels
    df_dataset["Golden"] = label_formating (df_dataset , label_type=label_type)

    # get data code dict ,   embedding matrix and the max length  of sentence
    train_embedding_weights , word2idx , max_length = embedding_data (df_dataset , PATH_W2V , EMBEDDING_DIM=50)
    embedding_weights , char2idx , max_len_word= char_embedding(df_dataset)

    # encode the dataset folowing the code dict and padding the sentences
    X = df_dataset["Review"].apply (lambda x: encode_data (word2idx , x , max_length))
    X_CHAR=  df_dataset["Review"].apply (lambda x: padding_after_encode(encode_data_char (x,char2idx ,  max_len_word),max_length,max_len_word))

    return  np.stack(X.values, axis=0) , np.stack(X_CHAR.values, axis=0) , np.stack (df_dataset["Golden"].values , axis=0)




def load_dataset_predict(path , word2idx ,char2idx, max_length,max_len_word):
    # read the csv
    # df_dataset = pd.read_csv (path , sep=";" , encoding="utf-8")

    # read excel
    df_dataset = pd.read_excel(path, encoding="utf8", errors='ignore')

    # delete no ascii chars, tokenize, encode
    df_dataset['Review'] = df_dataset["Review"].apply (
        lambda x: ''.join ([" " if ord (i) < 32 or ord (i) > 126 else i for i in x]))
    df_dataset["Review"] = df_dataset["Review"].apply (word_tokenize)
    X = df_dataset["Review"].apply (lambda x: encode_data (word2idx , x , max_length))
    X_CHAR=  df_dataset["Review"].apply (lambda x: padding_after_encode(encode_data_char (x,char2idx ,  max_len_word),max_length,max_len_word))

    # return the encoded data en numpy array
    return np.stack (X.values , axis=0), np.stack (X_CHAR.values , axis=0)


# formating label from positive neutral negative to label tyope
# @:param:
#       df_dataset: the dataset
#       label_type: the type of label encoding normal for simple classes and binary for onehote
# @:return
#       encoded labels
def label_formating(df_dataset , label_type="normal"):
    if label_type == "normal":

        df_dataset["Golden"] = df_dataset["Golden"].apply (
            lambda x: 2 if x == "Positive" else 0 if x == "Negative" else 1)

    else:

        if label_type == "binary":
            df_dataset["Golden"] = df_dataset["Golden"].apply (
                lambda x: [1 , 0 , 0] if x == "Positive" else [0 , 0 , 1] if x == "Negative" else [0 , 1 , 0])

    return df_dataset["Golden"]


# inverse the formating of labels from label type to positive neutral negative
def label_invers_formating(Y , label_type="normal"):
    if label_type == "normal":
        convert = (lambda Y: ["Positive" if x == 2 else "Negative" if x == 0 else "Neutral" for x in Y])
        return convert (Y)

    else:
        if label_type == "binary":
            to_int = lambda Y: [[1 if x >= 0.5 else 0 for x in z] for z in Y]
            Y = to_int (Y)
            convert = lambda Y: ["Positive" if x == [1 , 0 , 0] else "Negative" if x == [0 , 0 , 1] else "Neutral" for x
                                 in Y]
            return convert (Y)


# split dataset to training and test
def split_dataset(df_dataset):
    df_dataset.columns = ["Id" , "Review" , "Golden"]

    # shufll dataset
    df_dataset = df_dataset.sample (frac=1).reset_index (drop=True)

    # split the dataset
    X_train , X_test , Y_train , Y_test = train_test_split (df_dataset["Review"].values , df_dataset["Golden"].values)

    return X_train , X_test , Y_train , Y_test

def char_embedding(df_dataset, path_save_weights="../log/weights_char",path_save_words="../log/chars", EMBEDDING_DIM=5):
    # calculate max length sentece
    all_words = [word for tokens in df_dataset["Review"] for word in tokens]
    chars_dict=[]
    words_len= []
    for word in all_words:
        words_len.append (len (word))
        for char in word:
            if char not in chars_dict:
                chars_dict.append(char)


    # get training vocab char
    TRAINING_VOCAB = sorted (list (set (chars_dict)))


    print ("%s char total, with a vocabulary size of %s" % (len (chars_dict) , len (TRAINING_VOCAB)))
    print ("Max word length is %s" % max (words_len))

    # create char dictionnary
    char2idx = {w: idx for (idx , w) in enumerate (TRAINING_VOCAB)}
    idx2char = {idx: w for (idx , w) in enumerate (TRAINING_VOCAB)}

    # create dictinnary to map with word vector

    # initialize the embedding matrix vector
    embedding_weights = np.zeros ((len (char2idx) + 1 , len (char2idx) + 1))

    # search the word in the model, if found add its vector to the matrix, else, initialize it randomly
    chars = []
    for char , index in char2idx.items ():
        chars.append (char)
        embedding_weights[index , index] = 1

    # save the matrix and the word dict

    with open (path_save_weights , 'wb') as f:
        pickle.dump (embedding_weights , f)
    with open (path_save_words , 'wb') as f:
        pickle.dump (char2idx , f)


    return embedding_weights , char2idx , max (words_len)


def embedding_data(df_dataset , path_word2vec , EMBEDDING_DIM=50 , path_save_weights="../log/weights" ,
                   path_save_words="../log/words"):
    # calculate max length sentece
    all_words = [word for tokens in df_dataset["Review"] for word in tokens]
    sentence_lengths = [len (tokens) for tokens in df_dataset["Review"]]

    # get training vocab
    TRAINING_VOCAB = sorted (list (set (all_words)))
    print ("%s words total, with a vocabulary size of %s" % (len (all_words) , len (TRAINING_VOCAB)))
    print ("Max sentence length is %s" % max (sentence_lengths))

    # crete word dictionnary
    word2idx = {w: idx for (idx , w) in enumerate (TRAINING_VOCAB)}
    idx2word = {idx: w for (idx , w) in enumerate (TRAINING_VOCAB)}

    # create dictinnary to map with word vector

    # load word2vec model
    word_2_vec_model = models.KeyedVectors.load_word2vec_format (path_word2vec , binary=True )

    # initialize the embedding matrix vector
    train_embedding_weights = np.zeros ((len (word2idx) + 1 , EMBEDDING_DIM))

    # search the word in the model, if found add its vector to the matrix, else, initialize it randomly
    words = []
    for word , index in word2idx.items ():
        words.append (word)
        train_embedding_weights[index , :] = word_2_vec_model[word][
                                             :EMBEDDING_DIM] if word in word_2_vec_model else np.random.rand (
            EMBEDDING_DIM)

    # save the matrix and the word dict

    with open (path_save_weights , 'wb') as f:
        pickle.dump (train_embedding_weights , f)
    with open (path_save_words , 'wb') as f:
        pickle.dump (word2idx , f)
    return train_embedding_weights , word2idx , max (sentence_lengths)


# pading sequence



# encode data following the dict
def encode_data(word2idx , data , max_length):
    item_idx = []
    for word in data:
        if word in word2idx:
            item_idx.append (word2idx[word])
        else:
            item_idx.append (len (word2idx))
    result = padding (item_idx , max_length , len (word2idx))
    return result

# padding sentence (make them all with the same length)
def padding(sentence , max_length , pad_char):
    new_sentence = []

    for i in range ((max_length)):
        if i < len (sentence):
            new_sentence.append (sentence[i])
        else:
            new_sentence.append (pad_char)
    return np.array (new_sentence)


# padding words (make them all with the same length)
def word_padding(word , max_length , pad_char):
    new_word = []

    for i in range (max_length):
        if i < len (word):
            new_word.append (word[i])
        else:
            new_word.append (pad_char)
    return np.array (new_word)




# encode data following the dict
def encode_data_char(sentence, char2idx  , max_length):
    sentence_idx = []

    for word in sentence:
        word_idx=[]
        for char in word:
          if char in char2idx:
              word_idx.append (char2idx[char])
          else:
              word_idx.append (len (char2idx))
        sentence_idx.append(word_padding (word_idx , max_length , len (char2idx)))
    return np.array(sentence_idx)


#padding after encode

def padding_after_encode(sentence, max_sentence_length, max_word_length):

    new_sentence = []

    for i in range (max_sentence_length):
        if i < len (sentence):
            new_sentence.append (sentence[i])
        else:
            new_sentence.append (np.zeros(max_word_length))
    return np.array (new_sentence)

def create_emb_layer(weight_matrix , non_trainable=False):
        num_embedding , embedding_dim = weight_matrix.shape
        emb_layer = nn.Embedding (num_embedding , embedding_dim , padding_idx=num_embedding - 1)
        emb_layer.weight.data.copy_ (torch.from_numpy (weight_matrix))
        return emb_layer , num_embedding , embedding_dim

"""
def create_char_emb_layer(weight_matrix , non_trainable=False):
    num_embedding , embedding_dim = weight_matrix.shape
    emb_layer = nn.Embedding (num_embedding , embedding_dim , padding_idx=num_embedding - 1)
    emb_layer.weight.data.copy_ (torch.from_numpy (weight_matrix))
    return emb_layer , num_embedding , embedding_dim"""
if __name__ == '__main__':
    print((load_dataset ("../data/dev/news.csv",
                   "../../../../models/GoogleNews-vectors-negative300.bin")))




    # (embedding_data(df, "D:/Users/wissam/Documents/These/these/papers_material/VLDB_submission/data/GoogleNews-vectors-negative300.bin") )
