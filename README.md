### Group-9 CSCI 635 - Intro to Machine Learning:

> **End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures**<br>
> Makoto Miwa, Mohit Bansal<br> > [http://www.aclweb.org/anthology/P/P16/P16-1105.pdf](http://www.aclweb.org/anthology/P/P16/P16-1105.pdf)

## Relation Classification using LSTMs on Sequences and Tree Structures

We implemented a architecture based on the paper [End-to-End Relation Extraction using LSTMs
on Sequences and Tree Structures](http://www.aclweb.org/anthology/P/P16/P16-1105.pdf). This recurrent neural network based model captures both word sequence and dependency tree substructure information by stacking bidirectional treestructured LSTM-RNNs on bidirectional sequential LSTM-RNNs. This allows our model to jointly represent both entities and relations with shared parameters in a single model.

Our model allows
joint modeling of entities and relations in a single
model by using both bidirectional sequential
(left-to-right and right-to-left) and bidirectional
tree-structured (bottom-up and top-down) LSTMRNNs.

## Model

The model mainly consists of three representation layers:
a embeddings layer, a word sequence based LSTM-RNN layer (sequence layer), and finally a dependency subtree based LSTM-RNN layer (dependency layer).

![Relation Classification Network](/img/lstm_tree.jpg)

### Embedding Layer

Embedding layer consists of words, part-of-speech (POS) tags, dependency relations.

### Sequence Layer

The sequence layer represents words in a linear sequence
using the representations from the embedding layer. We represent the word sequence in a sentence with bidirectional LSTM-RNNs.
The LSTM unit at t-th word receives the concatenation of word and POS embeddings as its input vector.

<p align="center">
  <img src="/img/lstm_seq.jpg">
</p>

Tree-structured LSTM-RNN's equations :

<p align="center">
  <img src="/img/lstm_tree_eq.jpg">
</p>
### Data

SemEval-2010 Task 8 defines 9 relation types between nominals and a tenth type Other when two nouns have none of these relations and no direction is considered.

- Learning rate = 0.001
- Learning rate decay = 0.96
- state size = 100
- lambda_l2 = 0.0001
- Gradient Clipping = 10
- Entity Detection Pretrained

We implemented the above paper in tensorflow version2 using python 3.10.9.

To extract the entities and understand the relations between them LSTM - Long Short Term Memory Recurrent Neural Networks are used.
drop_out factor.py has detailed comments to understand the code.

steps to build the project :

1. Clone the Repository.
2. The below requirements are essential to run the code. However, we request you download the dependencies mentioned in requirements.txt file.
   a. Python > = 3.10.4
   b. tensorflow == 2.12
   c. nltk
3. Run python Relation-Classification-using-Bidirectional-LSTM-Tree
4. For Exploratory data analysis use Data_analysis.ipynb in a jupyter notebook setting.
