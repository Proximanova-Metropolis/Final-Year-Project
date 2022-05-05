"""
Author: Wissam Maamar kouadri
"""

import sys

sys.path.append (".")
import numpy as np
import torch
from bert_utils import load_dataset
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

GPU_AVAILABLE = (torch.cuda.is_available ())
np.random.seed (0)
torch.manual_seed (0)


class bert_cnn (nn.Module):
    def __init__(self , bert, WIN_SIZES=[3 , 4 , 5] , NUM_FILTERS=100 , EMBEDDING_DIM=768 ,
                 input_len=None , dropout_prob=0.5 , FUNCTION=0 , num_classes=3 , mode="None"):
        super (bert_cnn , self).__init__ ()
        self.WIN_SIZES = WIN_SIZES
        self.bert =bert
        self.input_len = input_len
        if GPU_AVAILABLE:
            self.embedding = self.embedding.cuda ()

        # create conv blocs
        conv_blocks = []
        for win_size in WIN_SIZES:
            maxpool_kernel_size = self.input_len - win_size + 1
            conv1d = nn.Conv1d (in_channels=EMBEDDING_DIM , out_channels=NUM_FILTERS , kernel_size=win_size ,
                                stride=1)
            component = nn.Sequential (
                conv1d ,
                nn.ReLU () ,
                nn.MaxPool1d (kernel_size=maxpool_kernel_size)
            )
            if GPU_AVAILABLE:
                component = component.cuda ()
            conv_blocks.append (component)


        self.conv_blocks = nn.ModuleList (
            conv_blocks)
        self.fc = nn.Linear (NUM_FILTERS * len (WIN_SIZES) , num_classes)

    # forward propagation
    def forward(self , x):
        self.bert.eval ()

        #embedding X
        with torch.no_grad ():
            tokens_tensor = torch.from_numpy (np.array(x[0])).long()
            segments_tensors = torch.from_numpy (np.array(x[1])).long()
            encoded_layers , _ = self.bert (tokens_tensor, segments_tensors)
            token_embeddings = torch.stack (encoded_layers , dim=0)
            token_embeddings = token_embeddings.permute (1 , 2 , 0,3)
            batch_embedding=[]
            for sentence in token_embeddings:
                token_vecs_sum = []
                for token in sentence:
                    # `token` is a [12 x 768] tensor

                    # Sum the vectors from the last four layers.
                    sum_vec = torch.sum (token[-4:] , dim=0)

                    token_vecs_sum.append (sum_vec)
                b = torch.Tensor (1, len( token_vecs_sum),  token_vecs_sum[0].size()[0])

                torch.cat (token_vecs_sum , out=b)

                batch_embedding.append(b.reshape(1,len( token_vecs_sum),  token_vecs_sum[0].size()[0]))


        b = torch.Tensor (len(token_embeddings),len (token_vecs_sum) , token_vecs_sum[0].size ()[0] )
        torch.cat (batch_embedding , out=b)
        x=b.reshape(len(token_embeddings),len (token_vecs_sum) , token_vecs_sum[0].size ()[0] )
        x = x.transpose (1 , 2)  #  convert x to (batch, embedding_dim, sentence_len)

        x_list = [conv_block (x) for conv_block in self.conv_blocks]
        out = torch.cat (x_list , 2)
        out = out.view (out.size (0) , -1)
        feature_extracted = out
        out = F.dropout (out , p=0.5 , training=self.training)
        return F.softmax (self.fc (out) , dim=1) , feature_extracted


if __name__ == '__main__':
    x1,x2,_ = load_dataset ("../data/dev/news.csv")
    bert= BertModel.from_pretrained ('bert-base-uncased')
    bert_cnn=bert_cnn (bert,  input_len= len(x1[0]), FUNCTION="softmax")
    bert_cnn.forward([x1,x2])