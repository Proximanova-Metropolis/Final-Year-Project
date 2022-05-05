# Quality-of-sentiment-analysis-tools



# Description 

This project is the material for the paper: Data quality in sentiment analysis tools: The reasons of inconsistency
The project contains implementations of the state-of-the-art sentiment analysis algorithms, the code of the experiment, the results, and information about the datasets used.



# Project-structure

## Code
This directory contains the codes of the project. It is structured as follows:
* `Sentiment_analysis_tools`  contains the six sentiment analysis tools evaluated in the study:
  * Vader:  allows to do sentiment analysis using the word-lexicon Vader 
  * senticnet5:  allows doing sentiment analysis using the concept-lexicon Senticnet 
  * sentiwordnet:  allows to do sentiment analysis using the word-lexicon sentiwordnet 
  * rec_nn:  allows to do sentiment analysis using the recursive deep models for semantic compositionality 
  * cnn_text: allows to do sentiment analysis using Kim's CNN with word embedding from word2vec-GoogleNews-vectors or Glove
  * char_cnn: allows to do sentiment analysis with CNN that use two embedding types: word2vec embedding +charachter embedding
  * bert_cnn: allows to do sentiment analysis using embedding from pre-trained Bert and Kim's CNN 
* `calculate_inconsistency.py`:  allows to calculate the inconsistency in different sentiment analysis tools
* `hyperparamaters_inc.py`:  calculate inconsistency resulted from different tunings for  CNN-text
* `normalize_dataset.py`: allows to unify the structure of datasets and logs
* `quality_verification.py`: represents the implementation of the  heuristics we use to enhance the quality of the generated dataset
 ## Data 
 Data are in the following drive: 
https://drive.google.com/drive/folders/1jdpZtsz06CY6FtYVbvmK_BvrsQc_FIUL?usp=sharing

This directory contains datasets of sentiment analysis extended with paraphrases. Each dataset is a list of analogical sets that were constructed using the generation method mentioned in "Adversarial Example Generation with Syntactically Controlled Paraphrase Networks" and has the following structure: 
 
 - Id: the Id of the Review. 
 - Review: the content of the Review.
 - Golden: the ground truth. It is the score attributed to the Review by human labeling.
 
NB: Semantically equivalent Reviews have the same Id.


#### Polarity labels 

The different polarity labels in different datasets are attributed as follows: 
* Amazon
     * negative : 0<= Golden <3
	 * neutral : 3
	 * positive : 3<golden <=5 
	 
* Sentiment Treebank 
   * negative : 0 <= golden <= 0.4
   * neutral : 0.4 < golden <= 0.6
   *  positive : 0.6< golden <= 1 
	 
* News headlines dataset
        * negative : -1 <= golden < 0
	* neutral : golden = 0
	* positive : 0< golden <= 1 

* First GOP debate

 It has three polarity labels: positive, negative, and neutral
 
 * US airlines tweets
 
 It has three polarity labels: positive, negative, and neutral

 
 NB: 	 
*  Data in the folder "sentiment_dataset_original" represents the original sentiment datasets.
* The folder "sentiment_dataset_with_labels" contains the datasets after converting sentiment score to labels.
*  The folder "sentiment_dataset_augmented" contains the augmented datasets before cleaning 
(data of this folder will e added after acceptation)
* The folder "clean sentiment dataset" includes the final clean datasets
(data of this folder will e added after acceptation)


## Experiments


This package is organized as follow: 
* `scripts` contains all the scripts of our experiments 
* `logs` contains  the logs of our experiments
* `plots` includes different plots of the experiments





# Program Usage


## Requirements 
This project was developed on Python 3.6.6. 
To install the requirements, run the following command: 

pip install requirement.txt

1- To apply sentiment analysis tools to your data:

python `method'.py --input_path input_path.csv  --out_path output_path.csv

`--input_path`  the path to the input data.

`--out_path`  the path to save the results.

`method` Vader, Sentiwordnet, Senticnet, rec_nn
2- To apply machine learning methods to your data

* Download a pre-trained model
* Download the word2vec pre-trained model  (word2vec-GoogleNews-vectors or Glove)
 
 ### Example

python text_cnn_predict.py --input_path input_path.csv   --save_path  model_path.pl --w2v_path word2vec_file  --embedding_type   --out_path output_path.csv

`--input_path` the path to the input data.
`--save_path  ` the path to the model 
`--w2v_path ` the path to the pre-trained w2v.
`--embedding_type` the embedding type (Gnews, Glove)
`--out_path` path to save the resulted log

3- To train the CCNs using other datasets



python text_cnn_train.py --input_path  training_data.csv --save_path models_dir --w2v_path word2vec_file --embedding_type  Gnews

4- To reproduce experimental results  : 
* Generate the logs 
   - For inconsistency experiments 

 python calculate_inconsystency.py --input_path path_to_input_data  --out_path path_to_save_logs  --function  exp
 

`--function` The experiment that we want to generate its logs (intool_inconsistency, intertool_inconsistency, intool_inconsistency_type, intertool_inconsistency_type,  intool_inc_polar_fact, intool_inconsistency_cos_wmd).
   *  For hyperparameters tuning and inconsistency experiments 

  
 python hyperparameters_inc.py --input_path --out_path  --function 
 
*  For time experiments 

python  time_exp.py --input_path path_to_input_data  --out_path path_to_save_logs
 
* Generate the plots

 python 'exp'.py --log_path  path_to_data_logs   
 

`log_path` path to the experiment logs
exp: the experiment we want to run. The experiments are in the folder Experiments/script



