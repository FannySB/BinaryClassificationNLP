
import numpy as np

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class NaiveBayes():
    def __init__(self, train_data, train_label):
        print('MultinomialNB fit...')
        self.text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB()),
        ])
        
        # get worst results using: stop_words='english'
        # self.text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
        #                     ('tfidf', TfidfTransformer()),
        #                     ('clf', MultinomialNB()),
        # ])

        
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                    'tfidf__use_idf': (True, False),
                    'clf__alpha': (1e-2, 1e-3),
        }


        self.gs_clf = GridSearchCV(self.text_clf, parameters, n_jobs=-1, cv=None)
        
        self.gs_clf = self.gs_clf.fit(train_data, train_label)

    def predict(self, test_data):
        return self.gs_clf.predict(test_data)

    def loss_fct(self, predicted, labels):
        return np.mean(predicted == labels)

class SVM():
    def __init__(self, train_data, train_label):
        print('SVM fit...')
        self.text_clf_svm = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                alpha=1e-3, max_iter =5, random_state=42, tol=None)),
        ])

        parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                    'tfidf__use_idf': (True, False),
                    'clf-svm__alpha': (1e-2, 1e-3),
        }

        
        self.gs_clf = GridSearchCV(self.text_clf_svm, parameters_svm, n_jobs=-1, cv=None)
        
        self.gs_clf = self.gs_clf.fit(train_data, train_label)

    def predict(self, test_data):
        return self.gs_clf.predict(test_data)

    def loss_fct(self, predicted, labels):
        return np.mean(predicted == labels)