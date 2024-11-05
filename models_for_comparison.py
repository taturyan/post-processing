import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from data_prep import get_adult_data, get_german_data, get_frequencies
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
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


########## evaluating fairlearn ##########
def get_fl_loss(tp, y, result_weights):
    num_h = len(result_weights)
    loss_list = [(tp.iloc[:, i] != y).astype(int) for i in range(num_h)]
    df = pd.concat(loss_list, axis=1)
    weighted_loss_vec = pd.DataFrame(np.dot(df, pd.DataFrame(result_weights)))
    loss_vec = weighted_loss_vec.iloc[:, 0]
    return loss_vec.mean()

def get_fl_predictions(fairlearn_clf, X):
    weights = fairlearn_clf.weights_[fairlearn_clf.weights_>0]
    hs = fairlearn_clf.predictors_[fairlearn_clf.weights_>0]
    pred_list = [pd.Series(h.predict(X)) for h in hs]
    total_pred = pd.concat(pred_list, axis=1, keys=range(len(weights)))
    return total_pred, weights

def extract_group_pred(total_pred, S):
    groups = sorted(list(pd.Series.unique(S)))
    pred_per_group = {}
    for g in groups:
        pred_per_group[g] = total_pred[S == g]
    return pred_per_group

def get_histogram(pred, _indices):
    hist = pd.Series(np.zeros(len(_indices)))
    for _index in _indices:
        hist[_indices == _index] = len(pred[pred == _index])
    return hist

def weighted_pmf(pred, classifier_weights, bins=[0,1]):
    _indices = pd.Series(bins)
    weights = list(classifier_weights)
    weighted_histograms = [(get_histogram(pred.iloc[:, i],_indices)) * weights[i]
                           for i in range(pred.shape[1])]
    _counts = sum(weighted_histograms)
    pmf = _counts / sum(_counts)
    return pmf

def pmf2disp(pmf1, pmf2):
    cdf_1 = pmf1.cumsum()
    cdf_2 = pmf2.cumsum()
    diff = cdf_1 - cdf_2
    diff = abs(diff)
    return max(diff)

def evaluate_fairlearn(dataset, num, eps_list, print_details = True,
            TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2, data_scaling=True, partial_training=False):
    #getting data
    if dataset=='adult':
        X, S, y = get_adult_data(problem='DP_unaware', as_df=True)
        #we take only 2000 samples for comparison
        sample_size = 2000 
        X, _, S, _, y ,_ = train_test_split(X, S, y, train_size = sample_size,  stratify=S, random_state=42)
        S_num = 2
        #p = get_frequencies(S)
    elif dataset=='german':        
        X, S, y = get_german_data(problem='DP_unaware', as_df=True)
        S_num = 2
        #p = get_frequencies(S)
    else:
        raise Exception('Dataset not found.')

    #scaling
    if data_scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cols=X.columns
        X = pd.DataFrame(X_scaled, columns=cols)
        
    #initialization 
    time_hist = []
    risk_all = {'mean':[], 'std':[]}
    unf_all = {}
    for s in range(S_num):
        unf_all[s] = {'mean':[], 'std':[]}
    ####################################

    for k, eps in enumerate(eps_list):
        if print_details:
            print (k+1,'/',len(eps_list), ' : collecting statistics for eps='+str(eps))
        
        risk = []
        unf = {}
        for s in range(S_num):
            unf[s] = []
     
        for i in range(1, num+1):

            X_df, X_test, S_df, S_test, y_df, y_test = train_test_split(X, S, y,
                                                                           test_size=TEST_SIZE, stratify=S, random_state=i)
            X_df.index, S_df.index, y_df.index = range(len(X_df)), range(len(S_df)), range(len(y_df))
            X_test.index, S_test.index, y_test.index = range(len(X_test)), range(len(S_test)), range(len(y_test))

            #additionally splitting into train and unlab according to our method
            if partial_training:
                X_train, X_unlab, S_train, S_unlab, y_train, y_unlab = train_test_split(X_df, S_df, y_df, 
                                                                            train_size = TRAIN_SIZE/(1-TEST_SIZE), stratify=S_df,
                                                                            random_state=i)
                X_train.index, S_train.index, y_train.index = range(len(X_train)),range(len(S_train)),range(len(y_train))
            else:
                X_train, S_train, y_train = X_df, S_df, y_df
    
            #training fairlearn
            start = time.time()

            base_clf = LogisticRegression(random_state=i)
            constraint = DemographicParity(difference_bound=eps)
            FL_clf = ExponentiatedGradient(base_clf, constraints=constraint)
            FL_clf.fit(X_train, y_train, sensitive_features=S_train)
            
            end = time.time()
            time_hist.append(end-start)
            
            #evaluation
            total_pred, weights = get_fl_predictions(FL_clf, X_test)
            pred_group = extract_group_pred(total_pred, S_test)

            PMF_all = weighted_pmf(total_pred, weights)
            PMF_group = [weighted_pmf(pred_group[g], weights) for g in pred_group]

            risk.append(get_fl_loss(total_pred, y_test, weights))
            for s in range(S_num):
                unf[s].append(pmf2disp(PMF_group[s], PMF_all))
            
            if print_details:    
                print ('-----   ', i,'/',num,': ADW: training completed; training time: ',end-start)

        risk_all['mean'].append(np.mean(risk))
        risk_all['std'].append(np.std(risk))
                
        for s in range(S_num):
            unf_all[s]['mean'].append(np.mean(unf[s]))
            unf_all[s]['std'].append(np.std(unf[s]))

        print ('---------------------------------------------------------')
        
     
    results = {'risk':risk_all,
               'unf':unf_all,
               'training_time_hist':time_hist}
            
    return results