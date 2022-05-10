import sys
sys.path.append (".")
import numpy as np
import torch
import char_cnn_utils
import char_cnn_model
from torch.autograd import Variable

from char_cnn_utils import load_dataset , create_emb_layer , embedding_data
import torch.nn as nn
import torch.nn.functional as F
import pickle
GPU_AVAILABLE = (torch.cuda.is_available ())
np.random.seed (0)
torch.manual_seed (0)


def predict (input_path,word2idx,char2idx, save_path= "../model/weights9.pt"):
    #load the model
    model = (torch.load (save_path))
    model.eval ()
    #def load_dataset_predict(path , word2idx , max_length,char2idx,max_len_word):

    #load and parse the data
    X , X_CHAR= char_cnn_utils.load_dataset_predict (input_path , word2idx ,char2idx, model.input_len, model.char_input_len)
    X = X.astype (int)
    X_CHAR=X_CHAR.astype (int)
    X = torch.from_numpy (X).long ()
    X_CHAR=torch.from_numpy (X_CHAR).long ()

    #prediction
    input_word, input_char = Variable (X),Variable(X_CHAR)
    preds , vector = model ([input_word,input_char])
    preds = torch.max (preds , 1)[1]
    pred_y = preds.data
    return pred_y


if __name__ == '__main__':


    #    return np.stack (X.values , axis=0) , X_CHAR.values , np.stack (df_dataset["Golden"].values , axis=0)

    X, X_char, Y = load_dataset ("../data/dev/news.csv",
                   "../../../../data/models/GoogleNews-vectors-negative300.bin")
    with open ("../log/weights" , 'rb') as f:
        pretrained_embeddings = pickle.load (f)
    with open ("../log/weights_char" , 'rb') as f:
        pretrained_embeddings_chars = pickle.load (f)


    max_len_sentence= X_char[1].shape[0]
    max_len_char = X_char[1].shape[1]
    cnn= char_cnn_model.char_cnn (26,weight_matrix= pretrained_embeddings , input_len=max_len_sentence ,char_input_len=max_len_char,char_weight_matrix= pretrained_embeddings_chars, out_embedding=10)
    print ("\n{}\n".format (str (cnn)))
    input=[]
    input.append(torch.from_numpy (np.array(X)).long ())
    input.append (torch.from_numpy(np.array(X_char)).long())
    cnn.forward(input)
