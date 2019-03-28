
import numpy as np
import math
import random

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

import data_config as config


class LoadData():
    def __init__(self, file):
        # Gather data
        print('Load data...')
        file_reader = open(file, 'r+', encoding='utf-8')
        self.datas = []
        elements = []
        for line in file_reader:
            line = {
                "line": line[:-3], 
                "elements": line[:-3].split(' '), 
                "label":line[-2]
            }
            elements += line['elements']
            self.datas.append(line)

        # Seed to keep Train and Valid separated, even if loaded separatly
        random.seed(config.shuffle_seed)
        random.shuffle(self.datas)
        self.count_lines = len(self.datas)

    def getTrainData(self):
        lines_train = []
        targets_train = []
        for cpt in range(math.floor(self.count_lines*config.train_perc)):
            lines_train.append(self.datas[cpt]['line'])
            targets_train.append(self.datas[cpt]['label'])
        return lines_train, targets_train


    def getTestData(self):
        lines_valid = []
        targets_valid = []
        for cpt in range(math.floor(self.count_lines*config.train_perc), self.count_lines):
            lines_valid.append(self.datas[cpt]['line'])
            targets_valid.append(self.datas[cpt]['label'])
        return lines_valid, targets_valid