"""
Author Wissam Mammar kouadri
"""

import bert_model
import bert_utils
import time
from sklearn.model_selection  import KFold
from  torch import nn, cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
import argparse
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

use_cuda= cuda.is_available()

def train(X1,X2,Y, cv , train_index , test_index, batch_size=5,WIN_SIZES=[3, 4, 5], NUM_FILTERS=100 ,nb_epoch=2,EMBEDDING_DIM=50, save_path= "../model/weights"):
    x1_train , x2_train, y_train = X1[train_index] ,X2[train_index] ,Y[train_index]
    x1_test , x2_test, y_test = X1[test_index] ,X2[test_index] , Y[test_index]
    #x_train=np.stack(x_train, axis=0)
    x1_train= x1_train.astype(int)
    x2_train = x2_train.astype (int)
    x1_train = torch.from_numpy (x1_train).long ()
    x2_train = torch.from_numpy (x2_train).long ()
    y_train = torch.from_numpy (y_train.astype(int)).long ()
    dataset_train = TensorDataset (x1_train , x2_train, y_train)
    # train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader = DataLoader (dataset_train , batch_size= batch_size , shuffle=True , num_workers=4 ,
                               pin_memory=False)

    x1_test = torch.from_numpy (x1_test).long ()
    x2_test = torch.from_numpy (x2_test).long ()
    y_test = torch.from_numpy (y_test).long ()
    if use_cuda:
        x1_test = x1_test.cuda ()
        y_test = y_test.cuda ()

    bert = BertModel.from_pretrained ('bert-base-uncased')
    #bert_cnn = bert_cnn (bert , input_len=len (x1[0]) , FUNCTION="softmax")

    # def __init__(self,WIN_SIZES=[3, 4, 5],NUM_FILTERS=100, EMBEDDING_DIM = 50,  weight_matrix=None, input_len=None, dropout_prob=0.5, FUNCTION=0, num_classes=3, mode="None"):
    model = bert_model.bert_cnn (bert , input_len= len(X1[1]),FUNCTION="softmax")
    if cv == 0:
        print ("\n{}\n".format (str (model)))

    if use_cuda:
        model = model.cuda ()

    parameters = filter (lambda p: p.requires_grad , model.parameters ())
    optimizer = torch.optim.Adam (parameters , lr=0.0002)

    loss_fn = nn.CrossEntropyLoss ()

    for epoch in range (nb_epoch):
        tic = time.time ()
        model.train ()
        for i , (x1,x2 , labels) in enumerate (train_loader):
            x1,x2 , labels = Variable (x1) , Variable (x2),Variable (labels)
            if use_cuda:
                x1,x2 , labels = x1.cuda () ,x2.cuda(), labels.cuda ()

            preds , _ = model ([x1,x2])
            if use_cuda:
                preds = preds.cuda ()
            loss = loss_fn (preds , labels)

            optimizer.zero_grad ()
            loss.backward ()
            optimizer.step ()



        model.eval ()
        eval_acc , sentence_vector = evaluate (model , [x1_test,x2_test] , y_test)
        # print('[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)'.format(epoch, loss.data[0], eval_acc, time.time()-tic) )
        print (
            '[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)'.format (epoch , loss.item () , eval_acc ,
                                                                                time.time () - tic))
        torch.save (model , save_path+str(epoch)+".pt")

    return eval_acc , sentence_vector

def evaluate(model , x , y_test):
        x1 = Variable (x[0])
        x2 = Variable (x[1])
        preds , vector = model ([x1,x2])
        preds = torch.max (preds , 1)[1]
        # eval_acc = sum(preds.data == y_test) / len(y_test)          # pytorch 0.3
        eval_acc = (preds.data == y_test).sum ().item () / len (y_test)  # pytorch 0.4
        return eval_acc , vector.cpu ().data.numpy ()

def main  (path_dataset, path_w2v, nb_epoch= 5, folds=10, save_path="", label_type=None, batch_size=5,WIN_SIZES=[3, 4, 5],NUM_FILTERS=100, EMBEDDING_DIM=50):
    cv_folds = folds
    kf = KFold (n_splits=cv_folds , shuffle=True , random_state=0)
    acc_list = []
    tic = time.time ()
    sentence_vectors , y_tests = [] , []
    X1,X2, Y= bert_utils.load_dataset(path_dataset,label_type= label_type)

    for cv , (train_index , test_index) in enumerate (kf.split (X1)):

        acc , sentence_vec = train (X1, X2, Y, cv , train_index , test_index, save_path=save_path, nb_epoch=nb_epoch,
                                    batch_size=batch_size,WIN_SIZES=WIN_SIZES , NUM_FILTERS=NUM_FILTERS,EMBEDDING_DIM=EMBEDDING_DIM)

        print ('cv = {}    train size = {}    test size = {}\n'.format (cv , len (train_index) , len (test_index)))
        acc_list.append (acc)
        sentence_vectors += sentence_vec.tolist ()
        y_tests += Y[test_index].tolist ()
    print ('\n avg acc = {:.3f}   (total time: {:.1f}s)\n'.format (sum (acc_list) / len (acc_list) , time.time () - tic))










if __name__ == "__main__":
    PROJECT_PATH = os.getcwd ()
    parser = argparse.ArgumentParser(description='Traning a Convolutional Neural Networks for polarity extraction in pytorch using bert')
    parser.add_argument ('--input_path' , type=str , default='../data/news.csv' ,
                         help='path to traning file, it should be a csv file')
    parser.add_argument ('--nb_epoch' , type=int , default=10 , help='Number of epochs ')
    parser.add_argument ('--batch_size' , type=int , default=5 , help='Mini batch size ')
    parser.add_argument ('--win_sizes' , type=int , nargs='*' , default=[3 , 4 , 5] ,
                         help='Windows sizes of filters default: [3, 4, 5]')
    parser.add_argument ('--num_filters' , type=int , default=100 ,
                         help='Number of filters in each window size default: 100')
    parser.add_argument ('--s' , type=float , default=3.0 , help='L2 norm constraint on w default: 3.0')
    parser.add_argument ('--dropout_prob' , type=float , default=0.5 , help='Dropout probability default: 0.5')
    parser.add_argument ('--label_type' , type=str , default="normal" , help='label encoding type (normal or binary for onehot default: normal')
    parser.add_argument ('--path_w2v' , type=str , default="../model/word2vec" , help='path to the word2vec model')
    parser.add_argument ('--folds' , type=int , default=5 , help='nb folds in croase validation')
    parser.add_argument ('--save_path' , type=str  , default="../model/" , help='path to save the pretrained model default ../model')
    parser.add_argument ('--embedding_dim' , type=int  , default=50 , help='embedding dim of the model')

    args = parser.parse_args ()


    main(args.input_path, args.path_w2v, nb_epoch=args.nb_epoch,folds=args.folds, save_path= args.save_path,label_type=args.label_type, batch_size=args.batch_size,
         WIN_SIZES=args.win_sizes, NUM_FILTERS=args.num_filters,EMBEDDING_DIM=  768  )