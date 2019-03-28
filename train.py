'''
    Inspired by https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
    DataSet used: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
'''
import numpy as np
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
    train_data = []
    train_label = []
    
    load_data = []
    for file in config.data_files:
        load_data.append(LoadData(file))

    for cpt in range(len(load_data)):
        train_x, train_y = load_data[cpt].getTrainData()
        train_data += train_x
        train_label += train_y

    nb_model_nb = NaiveBayes(train_data, train_label)
    nb_model_svm = SVM(train_data, train_label)

    # Save Naive Bayes Model
    nb_pickle = open(config.naive_bayes_path, 'wb')
    pickle.dump(nb_model_nb, nb_pickle)
    nb_pickle.close()

    # Save SVM Model
    svm_pickle = open(config.SVM_path, 'wb')
    pickle.dump(nb_model_nb, svm_pickle)
    svm_pickle.close()

    valid_data = []
    valid_label = []
    for cpt in range(len(load_data)):
        valid_x, valid_y = load_data[cpt].getTestData()
        valid_data += valid_x
        valid_label += valid_y
    predicted = nb_model_nb.predict(valid_data)
    print('results nb', nb_model_nb.loss_fct(predicted, valid_label))
    predicted = nb_model_svm.predict(valid_data)
    print('results svm', nb_model_svm.loss_fct(predicted, valid_label))
    


