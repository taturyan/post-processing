import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from data_prep import get_adult_data
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

#Denis,Hebiri
class BinClassRO:
    def __init__(self, base_classifier, alpha):
        self.base_classifier = base_classifier
        self.alpha = alpha #this corresponds to 1-eps from the paper

    def f(self, X):
        base_prob_pred = self.base_classifier.predict_proba(X)
        f_X = np.max(base_prob_pred, axis=1)
        return f_X

    def fit_CDF(self, f_X):
        return ECDF(f_X)

    def fit(self, X):
        base_pred_prob = self.base_classifier.predict_proba(X)
        f_X = self.f(X)
        self.CDF_f = self.fit_CDF(f_X)
        
    def predict(self, X):
        base_pred = self.base_classifier.predict(X)
        f_X = self.f(X)
        print(f_X)
        print(self.CDF_f(f_X))
        reject_mask = self.CDF_f(f_X) < self.alpha
        print(reject_mask)
        pred = np.where(reject_mask, 'R', base_pred)
        return pred

#test example
X, y = get_adult_data()
sample_size = 2000 
X, _, y ,_ = train_test_split(X, y, train_size=sample_size, random_state=0)

TRAIN_SIZE, UNLAB_SIZE, TEST_SIZE = 0.4, 0.4, 0.2

X_train, X_, y_train, y_ = train_test_split(X, y, train_size = TRAIN_SIZE, stratify = y)
X_unlab, X_test, y_unlab, y_test = train_test_split(X_, y_, test_size = TEST_SIZE/(1-TRAIN_SIZE), stratify = y_)

base_clf = LogisticRegression()
base_clf.fit(X_train, y_train)

RO_clf = BinClassRO(base_classifier=base_clf, alpha=0.05)
RO_clf.fit(X=X_unlab)
pred = RO_clf.predict(X_test)
print(pred)