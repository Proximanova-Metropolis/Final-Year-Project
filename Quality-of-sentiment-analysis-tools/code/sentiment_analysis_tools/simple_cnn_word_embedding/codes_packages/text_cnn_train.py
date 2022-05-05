"""
Author; Wissam Mammar kouadri
"""
import text_cnn_model
import text_cnn_utils
import time
from sklearn.model_selection  import KFold
from  torch import nn, cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
import argparse
import os
use_cuda= cuda.is_available()

def train(X,Y, cv , train_index , test_index, batch_size=5,WIN_SIZES=[3, 4, 5], NUM_FILTERS=100 ,nb_epoch=2,EMBEDDING_DIM=300, save_path= "../model/weights"):
    x_train , y_train = X[train_index] , Y[train_index]

    x_test , y_test = X[test_index] , Y[test_index]
    #x_train=np.stack(x_train, axis=0)
    x_train= x_train.astype(int)
    x_train = torch.from_numpy (x_train).long ()
    y_train = torch.from_numpy (y_train.astype(int)).long ()
    dataset_train = TensorDataset (x_train , y_train)
    # train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader = DataLoader (dataset_train , batch_size= batch_size , shuffle=True , num_workers=4 ,
                               pin_memory=False)

    x_test = torch.from_numpy (x_test).long ()
    y_test = torch.from_numpy (y_test).long ()
    if use_cuda:
        x_test = x_test.cuda ()
        y_test = y_test.cuda ()

    #load weights
    with open ("../log/weights" , 'rb') as f:
        pretrained_embeddings = pickle.load (f)


    # def __init__(self,WIN_SIZES=[3, 4, 5],NUM_FILTERS=100, EMBEDDING_DIM = 50,  weight_matrix=None, input_len=None, dropout_prob=0.5, FUNCTION=0, num_classes=3, mode="None"):
    model = text_cnn_model.text_cnn (WIN_SIZES=WIN_SIZES , NUM_FILTERS=NUM_FILTERS , EMBEDDING_DIM = EMBEDDING_DIM , weight_matrix=pretrained_embeddings, input_len= len(X[1]))
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
        for i , (inputs , labels) in enumerate (train_loader):
            inputs , labels = Variable (inputs) , Variable (labels)
            if use_cuda:
                inputs , labels = inputs.cuda () , labels.cuda ()

            preds , _ = model (inputs)
            if use_cuda:
                preds = preds.cuda ()
            loss = loss_fn (preds , labels)

            optimizer.zero_grad ()
            loss.backward ()
            optimizer.step ()

            """constrained_norm = 1
            if model.fc.weight.norm () > constrained_norm:
                model.fc.weight.data = model.fc.weight.data * constrained_norm / model.fc.weight.data.norm ()"""

        model.eval ()
        eval_acc , sentence_vector = evaluate (model , x_test , y_test)
        # print('[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)'.format(epoch, loss.data[0], eval_acc, time.time()-tic) )
        print (
            '[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)'.format (epoch , loss.item () , eval_acc ,
                                                                                time.time () - tic))
        torch.save (model , save_path+str(epoch)+".pt")

    return eval_acc , sentence_vector

def evaluate(model , x_test , y_test):
        inputs = Variable (x_test)
        preds , vector = model (inputs)
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
    X,Y= text_cnn_utils.load_dataset(path_dataset,path_w2v,label_type= label_type)

    for cv , (train_index , test_index) in enumerate (kf.split (X)):

        acc , sentence_vec = train (X,Y, cv , train_index , test_index,save_path=save_path, nb_epoch=nb_epoch,
                                    batch_size=batch_size,WIN_SIZES=WIN_SIZES,NUM_FILTERS=NUM_FILTERS,EMBEDDING_DIM=EMBEDDING_DIM)

        print ('cv = {}    train size = {}    test size = {}\n'.format (cv , len (train_index) , len (test_index)))
        acc_list.append (acc)
        sentence_vectors += sentence_vec.tolist ()
        y_tests += Y[test_index].tolist ()
    print ('\navg acc = {:.3f}   (total time: {:.1f}s)\n'.format (sum (acc_list) / len (acc_list) ,
                                                                  time.time () - tic))


    #np.save ('../model/sentence_vectors.npy' , np.array (sentence_vectors))
    #np.save ('../model/sentence_vectors_y.npy' , np.array (y_tests))







if __name__ == "__main__":
    PROJECT_PATH = os.getcwd ()
    parser = argparse.ArgumentParser(description='Traning a Convolutional Neural Networks for polarity extraction in pytorch')
    parser.add_argument ('--input_path' , type=str , default='../data/augmeted_dataset/news.csv' ,
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
         WIN_SIZES=args.win_sizes, NUM_FILTERS=args.num_filters,EMBEDDING_DIM=  300  )