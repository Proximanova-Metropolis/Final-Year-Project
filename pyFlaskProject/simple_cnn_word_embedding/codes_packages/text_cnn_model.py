"""

Author: Wissam mammar kouari
"""
import sys

sys.path.append (".")
import numpy as np
import torch
from text_cnn_utils import load_dataset , create_emb_layer , embedding_data
import torch.nn as nn
import torch.nn.functional as F

GPU_AVAILABLE = (torch.cuda.is_available ())
np.random.seed (0)
torch.manual_seed (0)


class text_cnn (nn.Module):
    def __init__(self , WIN_SIZES=[3 , 4 , 5] , NUM_FILTERS=100 , EMBEDDING_DIM=300 , weight_matrix=None ,
                 input_len=None , dropout_prob=0.5 , FUNCTION=0 , num_classes=3 , mode="None"):
        super (text_cnn , self).__init__ ()
        self.WIN_SIZES = WIN_SIZES
        self.embedding , _ , _ = create_emb_layer (weight_matrix)
        self.embedding.weight.requires_grad = mode == "nonstatic"
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
        x = self.embedding (x)  # shape (batch, max_sentence_len_embedding, embedding_dim)
        x = x.transpose (1 , 2)  #  convert x to (batch, embedding_dim, sentence_len)

        x_list = [conv_block (x) for conv_block in self.conv_blocks]
        out = torch.cat (x_list , 2)
        out = out.view (out.size (0) , -1)
        feature_extracted = out
        out = F.dropout (out , p=0.5 , training=self.training)
        return F.softmax (self.fc (out) , dim=1) , feature_extracted


if __name__ == '__main__':
    df = load_dataset (
        "../data/dev/news.csv")
    train_embedding_weights , word2idx = embedding_data (df ,
                                                         "../../../../data/GoogleNews-vectors-negative300.bin")
    text_cnn (weight_matrix=train_embedding_weights , input_len=10 , FUNCTION="softmax")