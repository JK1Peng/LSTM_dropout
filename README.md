### Group-9 CSCI 635 - Intro to Machine Learning:

## Relation Classification using LSTMs on Sequences and Tree Structures

We implemented a architecture based on the paper [End-to-End Relation Extraction using LSTMs
on Sequences and Tree Structures](http://www.aclweb.org/anthology/P/P16/P16-1105.pdf). This recurrent neural network based model captures both word sequence and dependency tree substructure information by stacking bidirectional treestructured LSTM-RNNs on bidirectional sequential LSTM-RNNs. This allows our model to jointly represent both entities and relations with shared parameters in a single model.

## Model

The model mainly consists of three representation layers:
a embeddings layer, a word sequence based LSTM-RNN layer (sequence layer), and finally a dependency subtree based LSTM-RNN layer (dependency layer).
Changes made :

1. Finetuned dropout factor
2. Finetuned Loss Metric
3. Modified the scheduled sampling
4. Modified the classifier from softmax to svm.

## Steps to build the project :

1. Clone the Repository.
2. The below requirements are essential to run the code. However, we request you download the dependencies mentioned in requirements.txt file.
   a. Python > = 3.10.4
   b. tensorflow == 2.12
   c. nltk
3. Run python Relation-Classification-using-Bidirectional-LSTM-Tree
4. For Exploratory data analysis use Data_analysis.ipynb in a jupyter notebook setting.
