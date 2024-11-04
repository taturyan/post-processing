import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from data_prep import get_adult_data, get_german_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from evaluation_measures import eval_risk, eval_risk_clf, eval_RejectOption, eval_RejectOption_clf

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
        f_X = self.f(X)
        self.CDF_f = self.fit_CDF(f_X)
        
    def predict(self, X):
        base_pred = self.base_classifier.predict(X)
        f_X = self.f(X)
        reject_mask = self.CDF_f(f_X) < self.alpha
        pred = np.where(reject_mask, 'R', base_pred)
        return pred
    
    def predict_proba(self, X):
        base_pred_prob = self.base_classifier.predict_proba(X)
        f_X = self.f(X)
        p_R = self.CDF_f(f_X)
        p_0_1 = base_pred_prob * (1 - p_R)[:, np.newaxis]
        return np.column_stack((p_0_1, p_R))

"""
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
pred = RO_clf.predict_proba(X_test)
#print(eval_risk_clf(pred, y_test))
#print(RO_clf.constraint_clf(pred))
"""

def evaluate_BinClassRO(dataset, num, alpha_list=[0.1, 0.05, 0.025], 
                            print_details = True, 
                            TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2, data_scaling=False):
    #getting data
    if dataset=='adult':
        X, y = get_adult_data()
        #we take only 2000 samples for comparison
        sample_size = 2000 
        X, _, y ,_ = train_test_split(X, y, train_size = sample_size,  stratify=y, random_state=42)
    elif dataset=='german':        
        X, y = get_german_data()
    else:
        raise Exception('Dataset not found.')

    #scaling
    if data_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
        # y = pd.Series(y_scaled.flatten(), index=y.index)

    #initialization 
    time_hist = []
    risk_all = {'mean':[], 'std':[]}
    risk_prob_all = {'mean':[], 'std':[]}
    base_risk, base_risk_all = {}, []
    base_risk_prob, base_risk_prob_all = {}, []
    constraint_all = {'mean':[], 'std':[]}
    constraint_prob_all = {'mean':[], 'std':[]}

    for k, alpha in enumerate(alpha_list):
        if print_details:
            print (k+1,'/',len(alpha_list), ' : collecting statistics for alpha='+str(alpha))
        
        risk, risk_prob = [], []
        constraint, constraint_prob = [], []
    
        for i in range(1, num+1):

            X_train, X_, y_train, y_ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=i)
            X_unlab, X_test, y_unlab, y_test = train_test_split(X_, y_, train_size = TRAIN_SIZE/(1-TEST_SIZE), 
                                                                stratify=y_,random_state=i)
            
            #training base classifier
            base_clf = LogisticRegression(random_state=i)
            base_clf.fit(X_train,y_train)

            #training BinClassRO
            start = time.time()
            BinClassRO_clf = BinClassRO(base_classifier=base_clf, alpha=alpha)
            BinClassRO_clf.fit(X=X_unlab)
            end = time.time()
            time_hist.append(end-start)
            
            #evaluation
            pred_prob_base = base_clf.predict_proba(X_test)
            pred_base = base_clf.predict(X_test)
            pred = BinClassRO_clf.predict(X_test)
            pred_prob = BinClassRO_clf.predict_proba(X_test)
            
            # #inverse scaling if needed
            # if data_scaling:
            #     y_test = scaler.inverse_transform(y_test.values.reshape(-1, 1))[:,0]

            risk.append(eval_risk_clf(pred, y_test))
            risk_prob.append(eval_risk(pred_prob, y_test, 'reject_option'))  
            base_risk_all.append(eval_risk_clf(pred_base, y_test))
            base_risk_prob_all.append(eval_risk(pred_prob_base, y_test, 'base'))
            constraint.append(eval_RejectOption_clf(pred))
            constraint_prob.append(eval_RejectOption(pred_prob))

            if print_details:    
                print ('-----   ', i,'/',num,': training completed, statistics collected')
        
        risk_all['mean'].append(np.mean(risk))
        risk_all['std'].append(np.std(risk))
        constraint_all['mean'].append(np.mean(constraint))
        constraint_all['std'].append(np.std(constraint))

        risk_prob_all['mean'].append(np.mean(risk_prob))
        risk_prob_all['std'].append(np.std(risk_prob))
        constraint_prob_all['mean'].append(np.mean(constraint_prob))
        constraint_prob_all['std'].append(np.std(constraint_prob))
            
        print ('---------------------------------------------------------')

    base_risk_prob['mean'] = np.mean(base_risk_prob_all)
    base_risk_prob['std'] = np.std(base_risk_prob_all)

    base_risk['mean'] = np.mean(base_risk_all)
    base_risk['std'] = np.std(base_risk_all)
            
    results = {'risk':risk_prob_all,
               'constraint':constraint_prob_all,
               'base_risk':base_risk_prob,
               'risk_clf':risk_all,
               'constraint_clf':constraint_all,
               'base_risk_clf':base_risk,
               'training_time_hist':time_hist}        
            
    return results
