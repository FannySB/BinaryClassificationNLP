import numpy as np


amazon_file = 'datas/amazon_cells_labelled.txt'
imbd_file = 'datas/imdb_labelled.txt'
yelp_file = 'datas/yelp_labelled.txt'
naive_bayes_path = 'saved_models/MultinomialNB.pkl'
SVM_path = 'saved_models/SGDClassifier.pkl'

data_files = [amazon_file, imbd_file, yelp_file]


train_perc = 0.80
test_perc = 1 - train_perc

shuffle_seed = 4
