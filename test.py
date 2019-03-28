'''
    Inspired by https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
    DataSet used: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
'''
import numpy as np
import math
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

import data_config as config
from load_data import LoadData
from models import SVM, NaiveBayes


if __name__ == "__main__":
    load_data = []
    for file in config.data_files:
        load_data.append(LoadData(file))

    # Open Naive Bayes Model
    nb_pickle = open(config.naive_bayes_path, 'rb')
    nb_model_nb = pickle.load(nb_pickle)
    nb_pickle.close()
    # Open SVM Model
    svm_pickle = open(config.SVM_path, 'rb')
    nb_model_svm = pickle.load(svm_pickle)
    svm_pickle.close()

    valid_data = []
    valid_label = []
    for cpt in range(len(load_data)):
        valid_x, valid_y = load_data[cpt].getTestData()
        valid_data += valid_x
        valid_label += valid_y
    predicted = nb_model_nb.predict(valid_data)
    print('results Naive Bayes accuracy', nb_model_nb.loss_fct(predicted, valid_label))
    predicted = nb_model_svm.predict(valid_data)
    print('results Support Vector Machine accuracy', nb_model_svm.loss_fct(predicted, valid_label))
    


