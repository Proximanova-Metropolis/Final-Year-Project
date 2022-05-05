"""
evaluate inconsistency in tools following different parameters

"""


# Author: Wissam  Mammar kouadri


import pandas as pd
import os
import argparse
import operator
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sentiment_analysis_tools.sentiwordnet import sentiment_analyzer_scores
from sentiment_analysis_tools.vader import sentiment_analyzer_scores

def polar_subjectif(document , golden):
    sentiment=max(sentiment_analyzer_scores(document).items(), key=operator.itemgetter(1))[0]


    if sentiment == "neu" and golden != "Neutral":
        return "Fact" , sentiment
    else:

        if sentiment == "neu" and golden == "Neutral":
            return "none" , sentiment
        else:
            if sentiment != "neu":
                return "Subjective" , sentiment
def euclidean_distance(corpus):



    vectorizer = CountVectorizer ()
    features = vectorizer.fit_transform (corpus).todense ()
    dist=euclidean_distances (features[0] , features[1])[0][0]

    return dist

def cos_sim(corpus):
    vectorizer = CountVectorizer (stop_words=None, token_pattern=r"(?u)\b\w+\b")
    features = vectorizer.fit_transform (corpus).todense ()
    dist=cosine_similarity (features[0] , features[1])[0][0]

    return dist


"""
Calculates the in-tool inconsistency in a data set 

Parameters
----------
input_path  :  string 
        the path to the dataset D that can be clustered to analogical sets
        the file should be a csv separated ";" with the structure "Id; Review, Golden, Pred"
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id ; Inc"
"""
def intool_inconsistency(input_path,out_path):

    #read csv
    df=pd.read_csv(input_path,sep=";")
    df_inc=pd.DataFrame(columns=["Inc"])

    #groupe by ID to get oly clusters
    clusters= df.groupby("Id")
    clusters= list(clusters)

    inc_cluster=[]
    Id=[]

    for cluster in clusters:
        inc_reviews = []
        if len((cluster[1]))>=2:
            Id.append (cluster[1].iloc[0]["Id"])
            for i_1 in range (len(cluster[1])):
                inc = 0
                for i_2 in range (len(cluster[1])):
                    if cluster[1].iloc[i_1]["Pred"]!=cluster[1].iloc[i_2]["Pred"]:
                         inc=inc+1
                inc=inc/len(cluster[1])

                #inc of the review
                inc_reviews.append((inc))
            inc_cluster.append(sum(inc_reviews)/len(inc_reviews))
    df_inc["Inc"]=inc_cluster
    df_inc["Id"]=Id
    df_inc.to_csv(out_path,sep=";",index=False)



"""
Calculates the inter-tool inconsistency between tools in a data set 


Parameters
----------
input_path  :  string 
        the path to the directory that contains sentiment analysis logs 
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id;pred1;pred2;..."
"""
def intertool_inconsistency(input_path,out_path):

    tools=["Pred1","pred2","Pred3","Pred4","Pred5","Pred6"]

    dirs=[join (input_path , o) for o in os.listdir (input_path) if os.path.isdir (os.path.join (input_path , o))]
    for dir in dirs:
        df=pd.DataFrame(columns=["Id","Pred1","pred2","Pred3","Pred4","Pred5","Pred6"])
        logs= [join (dir , o) for o in os.listdir (dir) if os.path.isfile (os.path.join (dir , o))]
        i=0
        for log in logs:
            df_log=pd.read_csv(log,sep=";")
            i=i+1

            df[df.columns[i]]= df_log["Pred"]
            df["Id"] = df_log["Id"]
        df_inc = pd.DataFrame (columns=["Id","Pred1","Pred2","Pred3","Pred4","Pred5","Pred6"])


        for i in range(len (df)):
            incs = []
            incs.append (df.iloc[i]["Id"])
            for tool_1 in tools:
                inc = 0
                for tool_2 in tools:
                   if df.iloc[i][tool_1] != df.iloc[i][tool_2]:
                        inc = inc + 1
                inc = inc /  (len (tools) - 1)
                incs.append(inc)
            df_inc.loc[i]=incs
        df_inc.to_csv(join(out_path,dir+"_intertool_inc.csv"),sep=";",index=False)


"""
Calculates the in-tool inconsistency  in a data set  and classify the results by type


Parameters
----------
input_path  :  string 
        the path to the dataset D that can be clustered to analogical sets
        the file should be a csv separated ";" with the structure "Id; Review, Golden, Pred"
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id; p_n; p_p; p_o; n_o; n_n; o_o "
"""
def intool_inconsistency_type(input_path,out_path):

    df=pd.read_csv(input_path,sep=";")
    df_inc=pd.DataFrame(columns=["p_n","o_n","p_o","p_p","n_n","o_o"])
    clusters= df.groupby("Id")
    clusters= list(clusters)
    inc_p_n_cluster=[]
    inc_o_n_cluster=[]
    inc_p_o_cluster=[]
    p_p_cluster=[]
    n_n_cluster=[]
    o_o_cluster=[]
    inc_cluster=[]

    for cluster in clusters:
        inc_p_n_reviews = []
        inc_o_n_reviews = []
        inc_p_o_reviews = []
        p_p_reviews = []
        n_n_reviews = []
        o_o_reviews = []
        inc_reviews=[]
        if len ((cluster[1])) >= 2:
            for i_1 in range (len(cluster[1])):
                inc_p_n = 0
                inc_o_n=0
                inc_p_o=0
                p_p=0
                n_n=0
                o_o=0
                inc =0

                for i_2 in range (len(cluster[1])):
                    if cluster[1].iloc[i_1]["Pred"] !=  cluster[1].iloc[i_2]["Pred"] :
                       inc=inc+1
                    if i_1 != i_2:

                        if cluster[1].iloc[i_1]["Pred"]=="Positive" and cluster[1].iloc[i_2]["Pred"]=="Negative":
                            inc_p_n = inc_p_n+1
                        if cluster[1].iloc[i_1]["Pred"] == "Negative"  and cluster[1].iloc[i_2]["Pred"] == "Positive":
                            inc_p_n = inc_p_n+1
                        if cluster[1].iloc[i_1]["Pred"]=="Neutral" and cluster[1].iloc[i_2]["Pred"]=="Negative":
                            inc_o_n = inc_o_n+1
                        if cluster[1].iloc[i_1]["Pred"] == "Negative"  and cluster[1].iloc[i_2]["Pred"] == "Neutral":
                            inc_o_n = inc_o_n+1
                        if cluster[1].iloc[i_1]["Pred"] == "Positive" and cluster[1].iloc[i_2]["Pred"] == "Neutral":
                            inc_p_o = inc_p_o + 1
                        if cluster[1].iloc[i_1]["Pred"] == "Neutral" and cluster[1].iloc[i_2]["Pred"] == "Positive":
                            inc_p_o = inc_p_o + 1
                        if cluster[1].iloc[i_1]["Pred"] == "Positive" and cluster[1].iloc[i_2]["Pred"] == "Positive":
                            p_p = p_p + 1
                        if cluster[1].iloc[i_1]["Pred"] == "Neutral" and cluster[1].iloc[i_2]["Pred"] == "Neutral":
                            o_o = o_o + 1
                        if cluster[1].iloc[i_1]["Pred"]=="Negative" and cluster[1].iloc[i_2]["Pred"]=="Negative":
                            n_n = n_n+1


                inc_p_n = inc_p_n / (len (cluster[1]))
                inc_o_n = inc_o_n / (len (cluster[1]))
                inc_p_o = inc_p_o / (len (cluster[1]))
                p_p = p_p / (len (cluster[1]))
                n_n = n_n / (len (cluster[1]))
                o_o = o_o / (len (cluster[1]))
                inc_p_n_reviews.append((inc_p_n))
                inc_o_n_reviews.append ((inc_o_n))
                inc_p_o_reviews.append ((inc_p_o))
                p_p_reviews.append ((p_p))
                n_n_reviews.append ((n_n))
                o_o_reviews.append ((o_o))
                inc = inc / len (cluster[1])
                inc_reviews.append ((inc))
            inc_cluster.append (sum (inc_reviews) / len (inc_reviews))
            inc_p_n_cluster .append(sum(inc_p_n_reviews)/len(inc_p_n_reviews))
            inc_o_n_cluster .append(sum(inc_o_n_reviews)/len(inc_o_n_reviews))
            inc_p_o_cluster .append(sum(inc_p_o_reviews)/len(inc_p_o_reviews))
            p_p_cluster .append(sum(p_p_reviews)/len(p_p_reviews))
            n_n_cluster .append(sum(n_n_reviews)/len(n_n_reviews))
            o_o_cluster .append(sum(o_o_reviews)/len(o_o_reviews))
    df_inc["Inc_p_n"]=inc_p_n_cluster
    df_inc["Inc_o_n"]=inc_o_n_cluster
    df_inc["Inc_p_o"]=inc_p_o_cluster
    df_inc["p_p"]=p_p_cluster
    df_inc["n_n"]=n_n_cluster
    df_inc["o_o"]=o_o_cluster
    df_inc.index.name="ID"
    df_inc.to_csv(out_path,sep=";")



"""

calculates the intertool inconsistency by type


Parameters
----------
input_path  :  string 
        the path to the directory that contains sentiment analysis logs 
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id; p_n; p_p; p_o; n_o; n_n; o_o "
"""


def intertool_inconsistency_type(input_path,out_path):

    tools=["Pred1","Pred2","Pred3","Pred4","Pred5","Pred6"]
    dirs=[join (input_path , o) for o in os.listdir (input_path) if os.path.isdir (os.path.join (input_path , o))]

    for dir in dirs:
        df=pd.DataFrame(columns=["Id","Pred1","Pred2","Pred3","Pred4","Pred5","Pred6"])


        logs= [join (dir , o) for o in os.listdir (dir) if os.path.isfile (os.path.join (dir , o))]
        i=0

        for log in logs:
            df_log=pd.read_csv(log,sep=";")
            i=i+1

            df[df.columns[i]]= df_log["Pred"]

            df = df[pd.notnull (df[tools])]
            df["Id"] = df_log["Id"]
        df_new=df

        for tool_1 in tools:

            df_inc = pd.DataFrame (columns=["Id" , "p_n" , "o_n" , "p_o" , "p_p" , "n_n" , "o_o"])
            incs_p_n = []

            incs_o_n = []

            incs_p_o =[]

            incs_p_p=[]

            incs_n_n=[]

            incs_o_o=[]


            for i in range (len (df_new)):
               inc_p_n = 0
               inc_o_n = 0
               inc_p_o = 0
               p_p = 0
               n_n = 0
               o_o = 0
               inc = 0
               for tool_2 in tools:
                   if tool_1 != tool_2:
                       if df_new.iloc[i][tool_1] == "Positive" and df_new.iloc[i][tool_2] == "Negative":
                           inc_p_n = inc_p_n + 1
                       if df_new.iloc[i][tool_1] == "Negative" and df_new.iloc[i][tool_2] == "Positive":
                           inc_p_n = inc_p_n + 1
                       if df_new.iloc[i][tool_1] == "Neutral" and df_new.iloc[i][tool_2] == "Negative":
                           inc_o_n = inc_o_n + 1
                       if df_new.iloc[i][tool_1] == "Negative" and df_new.iloc[i][tool_2] == "Neutral":
                           inc_o_n = inc_o_n + 1
                       if df_new.iloc[i][tool_1] == "Positive" and df_new.iloc[i][tool_2] == "Neutral":
                           inc_p_o = inc_p_o + 1
                       if df_new.iloc[i][tool_1] == "Neutral" and df_new.iloc[i][tool_2] == "Positive":
                           inc_p_o = inc_p_o + 1
                       if df_new.iloc[i][tool_1] == "Positive" and df_new.iloc[i][tool_2] == "Positive":
                           p_p = p_p + 1
                       if df_new.iloc[i][tool_1] == "Neutral" and df_new.iloc[i][tool_2] == "Neutral":
                           o_o = o_o + 1
                       if df_new.iloc[i][tool_1] == "Negative" and df_new.iloc[i][tool_2] == "Negative":
                           n_n = n_n + 1

                       if df_new.iloc[i][tool_1] != df_new.iloc[i][tool_2]:
                         inc = inc + 1
               incs_p_n.append( inc_p_n/(len (tools) - 1))
               incs_o_n.append(inc_o_n/(len (tools) - 1))
               incs_p_o.append( inc_p_o/(len (tools) - 1))
               incs_p_p.append(p_p/(len (tools) - 1))
               incs_n_n.append(n_n/(len (tools) - 1))
               incs_o_o.append(o_o/(len (tools) - 1))
            df_inc["Id"] = df_new["Id"]
            df_inc["Inc_p_n"]=incs_p_n
            df_inc[ "Inc_o_n"]=incs_o_n
            df_inc[ "Inc_p_o"] =incs_p_o
            df_inc["p_p"] =incs_p_p
            df_inc["n_n"] =incs_n_n
            df_inc["o_o"]= incs_o_o
            df_inc.to_csv (join (out_path , dir +"_"+ tool_1+ "_intertool_inc.csv") , sep=";" , index = False)
    df_inc.to_csv(join(out_path,dir+"_intertool_inc.csv"),sep=";",index=False)


"""
calculates the intool inconsistency as  a function of similarity mean of the cluster. 


Parameters
----------

input_path  :  string 
        the path to the dataset D that can be clustered to analogical sets
        the file should be a csv separated ";" with the structure "Id; Review, Golden, Pred"
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id; Inc; sim "
"""

def intool_inconsistency_sim(input_path,out_path):
    dirs = [join (input_path , o) for o in os.listdir (input_path) if os.path.isdir (os.path.join (input_path , o))]

    for dir in dirs:
        # get logs

        logs = [join (dir , o) for o in os.listdir (dir) if os.path.isfile (os.path.join (dir , o))]

        for log in logs:
            df=pd.read_csv(log,sep=";")
            df_inc=pd.DataFrame(columns=["Id","Golden","Inc","Dist"])
            clusters= df.groupby("Id")

            clusters= list(clusters)

            inc_cluster=[]
            sim_cluster=[]
            for cluster in clusters:
                inc_reviews = []
                for i_1 in range (len(cluster[1])):
                    inc = 0
                    for i_2 in range (len(cluster[1])):
                        if cluster[1].iloc[i_1]["Pred"] !=  cluster[1].iloc[i_2]["Pred"] :
                         inc=inc+1

                    inc = inc / len (cluster[1])

                    inc_reviews.append((inc))

                inc_cluster.append(sum(inc_reviews)/len(inc_reviews))
                sim_cluster.append(cluster[1]["Dist"].mean())
            df_inc["Inc"]=inc_cluster
            df_inc["Dist"]= sim_cluster
            df_inc.index.name = "Id"
            df_inc.to_csv(join(out_path, os.path.basename(log)),sep=";")



"""
Calculates the in-tool inconsistency  in a data set  and classify the results by the sentence parsing
to see the relation between tools inconsistency and the sentence parsing


Parameters
----------
input_path  :  string 
        the path to the dataset D that can be clustered to analogical sets
        the file should be a csv separated ";" with the structure "Id; Review, Golden, Pred"
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id; Inc; Parse "
"""



def intool_inc_parse(input_path,out_path):

    df=pd.read_csv(input_path,sep=";")
    df.columns=["Id","Review","Parse","Golden","Pred"]
    df_inc=pd.DataFrame(columns=["Id","Review","Parse","Golden","Pred","Inc"])
    clusters= df.groupby("Id")

    clusters= list(clusters)

    j=1;
    for cluster in clusters:
        inc_reviews = []
        if len((cluster[1]))>2:
            for i_1 in range (len(cluster[1])):
                inc = 0
                for i_2 in range (len(cluster[1])):
                    if cluster[1].iloc[i_1]["Pred"]!=cluster[1].iloc[i_2]["Pred"]:
                         inc=inc+1
                inc=inc/len(cluster[1])

                df_inc.loc[j]   = [cluster[1].iloc[i_1]["Id"],cluster[1].iloc[i_1]["Review"],
                                    cluster[1].iloc[i_1]["Parse"],cluster[1].iloc[i_1]["Golden"],
                                    cluster[1].iloc[i_1]["Pred"],inc]
                j=j+1

    df_inc.to_csv(out_path,sep=";")


"""
Calculates the in-tool inconsistency  in a data set  and classify the the document nature (polar fact or opinionated 
to see the relation between tools inconsistency and the document nature

Parameters
----------
input_path  :  string 
        the path to the dataset D that can be clustered to analogical sets
        the file should be a csv separated ";" with the structure "Id; Review, Golden, Pred"
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id; Review1,; Review2; Pred1; Pred2; Golden; Inc; Nature1; Nature2 "
"""
def intool_inc_polar_fact(input_path, out_path):
    df = pd.read_csv (input_path , sep=";")
    df_inc = pd.DataFrame (columns=["Id" , "Review1" ,"Review2",  "Pred1" , "Golden" , "Inc" , "Nature1" ,"Nature2"])

    clusters = df.groupby ("Id")
    clusters = list (clusters)
    Nature1 = []
    Nature2 = []
    incs = []
    reviews1=[]
    reviews2=[]
    golden=[]
    pred1=[]
    pred2=[]
    sw1=[]
    sw2=[]
    for cluster in clusters:

        if len ((cluster[1])) >= 2:
            for i_1 in range (len (cluster[1])):
                o1 , s1 = polar_subjectif (cluster[1].iloc[i_1]["Review"] , cluster[1].iloc[i_1]["Golden"])
                for i_2 in range (len (cluster[1])):
                    inc = 0
                    if cluster[1].iloc[i_1]["Pred"] != cluster[1].iloc[i_2]["Pred"]:
                        inc = 1
                    o2,s2=polar_subjectif (cluster[1].iloc[i_2]["Review"], cluster[1].iloc[i_2]["Golden"])
                    Nature1.append ( o1)
                    Nature2.append ( o2)
                    incs.append (inc)
                    reviews1.append (cluster[1].iloc[i_1]["Review"])
                    reviews2.append (cluster[1].iloc[i_2]["Review"])
                    golden.append (cluster[1].iloc[i_1]["Golden"])
                    pred1.append (cluster[1].iloc[i_1]["Pred"])
                    pred2.append (cluster[1].iloc[i_2]["Pred"])
                    sw1.append(s1)
                    sw2.append(s2)


    df_inc["Nature1"] = Nature1
    df_inc["Nature2"] = Nature2
    df_inc["Inc"] = incs
    df_inc["Review1"] = reviews1
    df_inc["Review2"] = reviews2
    df_inc["Golden"] = golden
    df_inc["Pred1"] = pred1
    df_inc["Pred2"] = pred2
    df_inc["Inc"] = incs
    df_inc["s1"] = sw1
    df_inc["s2"] = sw2
    #df_inc=df_inc.drop_duplicates(subset=["Review1","Review2"])
    df_inc.to_csv(out_path, sep=";",index=False)



"""
Calculates the in-tool inconsistency  in a data set  and classify the accuracy

Parameters
----------
input_path  :  string 
        the path to the dataset D that can be clustered to analogical sets
        the file should be a csv separated ";" with the structure "Id; Review, Golden, Pred"
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id; Inc; Acc "
"""
def inc_acc(input_path, out_path):
    df = pd.read_csv (input_path , sep=";")
    df_inc = pd.DataFrame (columns=["Inc","Acc"])
    clusters = df.groupby ("Id")
    clusters = list (clusters)
    inc_cluster = []
    accuracies=[]
    for cluster in clusters:
        inc_reviews = []
        accuracy=0
        if len ((cluster[1])) > 2:
            for i_1 in range (len (cluster[1])):
                inc = 0
                for i_2 in range (len (cluster[1])):
                    if cluster[1].iloc[i_1]["Pred"] != cluster[1].iloc[i_2]["Pred"]:
                        inc = inc + 1
                inc = inc / len (cluster[1])
                inc_reviews.append ((inc))
                if cluster[1].iloc[i_1]["Pred"] == cluster[1].iloc[i_1]["Golden"]:
                    accuracy=accuracy+1

            inc_cluster.append (sum (inc_reviews) / len (inc_reviews))
            accuracies.append(accuracy/len(cluster[1]))
    df_inc["Inc"] = inc_cluster
    df_inc["Acc"] = accuracies
    df_inc.index.name = "Id"
    df_inc.to_csv (out_path , sep=";")


"""
Calculates the in-tool inconsistency  in a data set  and classify cos, and wmd sim

Parameters
----------
input_path  :  string 
        the path to the dataset D that can be clustered to analogical sets
        the file should be a csv separated ";" with the structure "Id; Review, Golden, Pred"
out_path    :  string
        the path to save the log file 
        the file is a .csv separated with ";"  with the structure "Id; Review1; Review2; Cos_sim; Wmd_sim "
"""


def intool_inconsistency_cos_wmd(input_path,out_path):
    df = pd.read_csv (input_path , sep=";")
    df_inc = pd.DataFrame (columns=["Inc"])
    clusters = df.groupby ("Id")

    clusters = list (clusters)

    inc_cluster = []
    dif_cluster = []
    reviews1=[]
    reviews2=[]
    gold=[]
    pred1=[]
    pred2 = []
    for cluster in clusters:

        if len ((cluster[1])) > 2:
            for i_1 in range (len (cluster[1])):

                review1 =cluster[1].iloc[i_1]["Review"]
                for i_2 in range (len (cluster[1])):
                    review2 = cluster[1].iloc[i_2]["Review"]
                    dist= cos_sim([review1,review2])
                    inc = 0
                    if cluster[1].iloc[i_1]["Pred"] != cluster[1].iloc[i_2]["Pred"]:
                        inc =  1

                    inc_cluster.append (inc)
                    dif_cluster.append(dist)
                    reviews1.append(cluster[1].iloc[i_1]["Review"])
                    reviews2.append(cluster[1].iloc[i_2]["Review"])
                    gold.append(cluster[1].iloc[i_1]["Golden"])
                    pred1.append(cluster[1].iloc[i_1]["Pred"])
                    pred2.append(cluster[1].iloc[i_2]["Pred"])


    df_inc["Inc"] = inc_cluster
    df_inc["cos_sim"] = dif_cluster
    df_inc["Review1"]=reviews1
    df_inc["Review2"] = reviews2
    df_inc["Pred2"] = pred2
    df_inc["Pred1"] = pred1
    df_inc["Golden"]= gold
    df_inc.index.name = "ID"
    df_inc.to_csv (out_path , sep=";")


def intool_inconsistency_cos(input_path,out_path):
    df = pd.read_csv (input_path , sep=";")
    cosdist=[]
    for i in range(len(df)):
        cosdist.append(cos_sim([df.iloc[i]["Review1"],df.iloc[i]["Review2"]]))
    df["Cos_sim"]=cosdist
    df.to_csv (out_path , sep=";")




if __name__ == '__main__':
    # Ignore warning message by tensor flow
    # model args
    parser = argparse.ArgumentParser ()

    parser.add_argument ('--input_path' , type=str , default="./VLDB_submission/experiments/logs/logs_dev" ,
                                                              help=' lpath to input dataset')
    parser.add_argument ('--out_path' , type=str , default='./VLDB_submission/experiments/logs/logs_dev' ,
                         help='data save path')


    parser.add_argument("--function", type=str, default="intool_inconsistency" , help="Function")
    args = parser.parse_args()

    onlyfiles = [f for f in listdir (args.input_path) if isfile (join (args.input_path , f))]

    #inc_pars (args.input_path , args.out_path)
    if args.function == "intool_inconsistency":
        path_1 = join (args.out_path , 'intool_inconsistency')
        if not os.path.exists (path_1): os.mkdir (path_1)
        for file in onlyfiles:
                file_name , _ = os.path.splitext (file)
                if "stanfordtool" in file_name:
                    path_2= join(path_1, "rec_nn")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency (join(args.input_path,file) , join(path_2, file_name+'_inc_degree.csv'))
                if "char_cnn" in file_name:
                    path_2= join(path_1, "char_cnn")
                    if not os.path.exists (path_2): os.mkdir(path_2)
                    intool_inconsistency (join(args.input_path,file) , join(path_2, file_name+'_inc_degree.csv'))
                if "cnn_text" in file_name:
                    path_2= join(path_1, "text_cnn")
                    if not os.path.exists(path_2): os.mkdir(path_2)

                    intool_inconsistency (join(args.input_path,file) , join(path_2, file_name+'_inc_degree.csv'))
                if "senticnet" in file_name:
                    path_2= join(path_1, "senticnet")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency (join(args.input_path,file) , join(path_2, file_name+'_inc_degree.csv'))
                if "sentiwordnet" in file_name:
                    path_2= join(path_1, "sentiwordnet")
                    if not os.path.exists(path_2): os.mkdir(path_2)

                    intool_inconsistency (join(args.input_path,file) , join(path_2, file_name+'_inc_degree.csv'))
                if "vader" in file_name:
                    path_2= join(path_1, "vader")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency (join(args.input_path,file) , join(path_2, file_name+'_inc_degree.csv'))
                if "glove_cnn" in file_name:
                    path_2= join(path_1, "glove_cnn")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency (join(args.input_path,file) , join(path_2, file_name+'_inc_degree.csv'))
                if "besrt_cnn" in file_name:
                    path_2= join(path_1, "bert_cnn")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency (join(args.input_path,file) , join(path_2, file_name+'_inc_degree.csv'))
    if args.function == "intertool_inconsistency":
       path_1 = join (args.out_path , 'intertool_inconsistency')

       if not os.path.exists (path_1): os.mkdir (path_1)
       #file_name , _ = os.path.splitext (file)
       intertool_inconsistency ( args.input_path  , path_1)
    if args.function == "intertool_inconsistency_type":
        path_1 = join (args.out_path , "intertool_inconsistency_type")
        if not os.path.exists (path_1): os.mkdir (path_1)
        # file_name , _ = os.path.splitext (file)
        intertool_inconsistency_type (args.input_path , path_1)
    if args.function == "intool_inconsistency_cos":
        path_1 = join (args.out_path , 'intool_inconsistency_cos')


        if not os.path.exists (path_1): os.mkdir (path_1)
        for file in onlyfiles:
                file_name , _ = os.path.splitext (file)
                if "stanfordtool" in file_name:

                    path_2= join(path_1, "rec_nn")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos (join(args.input_path,file) , join(path_2, file_name+'_cos_inconsistency.csv'))
                if "char_cnn" in file_name:
                    path_2= join(path_1, "char_cnn")
                    if not os.path.exists (path_2): os.mkdir(path_2)
                    intool_inconsistency_cos (join(args.input_path,file) , join(path_2, file_name+'_cos_inconsistency.csv'))
                if "cnn_text" in file_name:
                    path_2= join(path_1, "text_cnn")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos (join(args.input_path,file) , join(path_2, file_name+'_cos_inconsistency.csv'))
                if "senticnet" in file_name:
                    path_2= join(path_1, "senticnet")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos (join(args.input_path,file) , join(path_2, file_name+'_cos_inconsistency.csv'))
                if "sentiwordnet" in file_name:
                    path_2= join(path_1, "sentiwordnet")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos (join(args.input_path,file) , join(path_2, file_name+'_cos_inconsistency.csv'))
                if "vader" in file_name:
                    path_2= join(path_1, "vader")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos (join(args.input_path,file) , join(path_2, file_name+'_cos_inconsistency.csv'))
    if args.function == "intool_inconsistency_cos_wmd":
        path_1 = join (args.out_path , 'intool_inconsistency_cos_wmd')
        if not os.path.exists (path_1): os.mkdir (path_1)
        for file in onlyfiles:
                file_name , _ = os.path.splitext (file)
                if "stanfordtool" in file_name:
                    path_2= join(path_1, "rec_nn")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos_wmd (join(args.input_path,file) , join(path_2, file_name+'_cos_wmd_inconsistency.csv'))
                if "char_cnn" in file_name:
                    path_2= join(path_1, "char_cnn")
                    if not os.path.exists (path_2): os.mkdir(path_2)
                    intool_inconsistency_cos_wmd (join(args.input_path,file) , join(path_2, file_name+'_cos_wmd_inconsistency.csv'))
                if "cnn_text" in file_name:
                    path_2= join(path_1, "cnn_text")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos_wmd (join(args.input_path,file) , join(path_2, file_name+'_cos_wmd_inconsistency.csv'))
                if "senticnet" in file_name:
                    path_2= join(path_1, "senticnet")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos_wmd (join(args.input_path,file) , join(path_2, file_name+'_cos_wmd_inconsistency.csv'))
                if "sentiwordnet" in file_name:
                    path_2= join(path_1, "sentiwordnet")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos_wmd (join(args.input_path,file) , join(path_2, file_name+'_cos_wmd_inconsistency.csv'))
                if "vader" in file_name:
                    path_2= join(path_1, "vader")
                    if not os.path.exists(path_2): os.mkdir(path_2)
                    intool_inconsistency_cos_wmd (join(args.input_path,file) , join(path_2, file_name+'_cos_wmd_inconsistency.csv'))
    if args.function == "intool_inconsistency_type":
        path_1 = join (args.out_path , 'intool_inconsistency_type')

        if not os.path.exists (path_1): os.mkdir (path_1)
        for file in onlyfiles:
            file_name , _ = os.path.splitext (file)

            if "stanfordtool" in file_name:

                path_2 = join (path_1 , "rec_nn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inconsistency_type (join (args.input_path , file) , join (path_2 , file_name + 'intool_inconsistency_type.csv'))
            if "char_cnn" in file_name:
                path_2 = join (path_1 , "char_cnn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inconsistency_type (join (args.input_path , file) , join (path_2 , file_name + 'intool_inconsistency_type.csv'))
            if "cnn_text" in file_name:
                path_2 = join (path_1 , "text_cnn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inconsistency_type (join (args.input_path , file) , join (path_2 , file_name + 'intool_inconsistency_type.csv'))
            if "senticnet" in file_name:
                path_2 = join (path_1 , "senticnet")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inconsistency_type (join (args.input_path , file) , join (path_2 , file_name + 'intool_inconsistency_type.csv'))
            if "sentiwordnet" in file_name:
                path_2 = join (path_1 , "sentiwordnet")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inconsistency_type (join (args.input_path , file) , join (path_2 , file_name + 'intool_inconsistency_type.csv'))
            if "vader" in file_name:
                path_2 = join (path_1 , "vader")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inconsistency_type (join (args.input_path , file) , join (path_2 , file_name + 'intool_inconsistency_type.csv'))
    if args.function == "intool_inc_parse":
        path_1 = join (args.out_path , 'intool_inc_parse')

        if not os.path.exists (path_1): os.mkdir (path_1)
        for file in onlyfiles:
            file_name , _ = os.path.splitext (file)

            if "stanfordtool" in file_name:

                path_2 = join (path_1 , "rec_nn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_parse (join (args.input_path , file) , join (path_2 , file_name + '_intool_inc_pars.csv'))
            if "char_cnn" in file_name:
                path_2 = join (path_1 , "char_cnn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_parse (join (args.input_path , file) , join (path_2 , file_name + '_intool_inc_pars.csv'))
            if "cnn_txt" in file_name:
                path_2 = join (path_1 , "text_cnn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_parse (join (args.input_path , file) , join (path_2 , file_name + '_intool_inc_pars.csv'))
            if "senticnet" in file_name:
                path_2 = join (path_1 , "senticnet")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_parse (join (args.input_path , file) , join (path_2 , file_name + '_intool_inc_pars.csv'))
            if "sentiwordnet" in file_name:
                path_2 = join (path_1 , "sentiwordnet")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_parse (join (args.input_path , file) , join (path_2 , file_name + '_intool_inc_pars.csv'))
            if "vader" in file_name:
                path_2 = join (path_1 , "vader")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_parse (join (args.input_path , file) , join (path_2 , file_name + '_intool_inc_pars.csv'))
    if args.function  == "intool_inc_polar_fact":
        path_1 = join (args.out_path , 'fact_inc')


        if not os.path.exists (path_1): os.mkdir (path_1)
        for file in onlyfiles:
            file_name , _ = os.path.splitext (file)

            if "stanfordtool" in file_name:

                path_2 = join (path_1 , "rec_nn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_polar_fact (join (args.input_path , file) , join (path_2 , file_name + 'fact_inc.csv'))
            if "char_cnn" in file_name:
                path_2 = join (path_1 , "char_cnn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_polar_fact (join (args.input_path , file) , join (path_2 , file_name + 'fact_inc.csv'))
            if "cnn_text" in file_name:
                path_2 = join (path_1 , "text_cnn")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_polar_fact (join (args.input_path , file) , join (path_2 , file_name + 'fact_inc.csv'))
            if "senticnet" in file_name:
                path_2 = join (path_1 , "senticnet")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_polar_fact (join (args.input_path , file) , join (path_2 , file_name + 'fact_inc.csv'))
            if "sentiwordnet" in file_name:
                path_2 = join (path_1 , "sentiwordnet")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_polar_fact (join (args.input_path , file) , join (path_2 , file_name + 'fact_inc.csv'))
            if "vader" in file_name:
                path_2 = join (path_1 , "vader")
                if not os.path.exists (path_2): os.mkdir (path_2)
                intool_inc_polar_fact (join (args.input_path , file) , join (path_2 , file_name + 'fact_inc.csv'))


