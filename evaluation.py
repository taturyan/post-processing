import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import time

from data_prep import get_adult_data, get_german_data, get_frequencies, get_communities_data
from models_for_comparison import BinClassRO
from PostProc import PostProc
from evaluation_measures import eval_risk, eval_risk_clf, eval_RejectOption, eval_RejectOption_clf, eval_DP

    
def evaluate_reject_option(dataset, num, K=2, T=5000, alpha_list=[0.1, 0.05, 0.025], 
                            print_details = True, beta='auto',
                            alg={'base':'SGD3', 'method':'ACSA'},
                            TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2, data_scaling=False):
    #getting data
    if dataset=='adult':
        X, y = get_adult_data()
        #we take only 2000 samples for comparison
        sample_size = 2000 
        X, _, y ,_ = train_test_split(X, y, train_size = sample_size,  stratify=y, random_state=42)
    elif dataset=='german':        
        X, y = get_german_data()
    elif dataset=='communities':        
        X, y = get_communities_data()
    else:
        raise Exception('Dataset not found.')

    #scaling
    if data_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    #initialization 
    time_hist = []
    risk_all = {'mean':[], 'std':[]}
    risk_argmax_all = {'mean':[], 'std':[]}
    base_risk, base_risk_all = {}, []
    base_risk_argmax, base_risk_argmax_all = {}, []
    constraint_all = {'mean':[], 'std':[]}
    constraint_argmax_all = {'mean':[], 'std':[]}

    for k, alpha in enumerate(alpha_list):
        if print_details:
            print (k+1,'/',len(alpha_list), ' : collecting statistics for alpha='+str(alpha))
        
        risk, risk_argmax = [], []
        constraint, constraint_argmax = [], []
    
        for i in range(1, num+1):

            X_train, X_, y_train, y_ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=i)
            X_unlab, X_test, y_unlab, y_test = train_test_split(X_, y_, train_size = TRAIN_SIZE/(1-TEST_SIZE), 
                                                                stratify=y_,random_state=i)
            
            #training base classifier
            base_clf = LogisticRegression(multi_class="multinomial", solver='lbfgs', 
                                           max_iter=5000, random_state=i)
            base_clf.fit(X_train,y_train)

            #training PostProc
            start = time.time()
            PostProc_clf = PostProc(problem='reject_option', base_classifier=base_clf, K=K, T=T,
                                    RejectOption_params = {'B':alpha})
            PostProc_clf.fit(X=X_unlab, beta =beta, alg=alg)
            end = time.time()
            time_hist.append(end-start)
            
            pred_base = base_clf.predict_proba(X_test)
            pred_argmax_base = base_clf.predict(X_test)
            pred = PostProc_clf.predict(X_test)
            pred_argmax = np.array(PostProc_clf.A)[np.argmax(pred, axis=1)]

            risk.append(eval_risk(pred, y_test, 'reject_option'))            
            base_risk_all.append(eval_risk(pred_base, y_test, 'base'))
            constraint.append(eval_RejectOption(pred))

            risk_argmax.append(eval_risk_clf(pred_argmax, y_test))            
            base_risk_argmax_all.append(eval_risk_clf(pred_argmax_base, y_test))
            constraint_argmax.append(eval_RejectOption_clf(pred_argmax))

            if print_details:    
                print ('-----   ', i,'/',num,': training completed, statistics collected')
        
        risk_all['mean'].append(np.mean(risk))
        risk_all['std'].append(np.std(risk))
        constraint_all['mean'].append(np.mean(constraint))
        constraint_all['std'].append(np.std(constraint))

        risk_argmax_all['mean'].append(np.mean(risk_argmax))
        risk_argmax_all['std'].append(np.std(risk_argmax))
        constraint_argmax_all['mean'].append(np.mean(constraint_argmax))
        constraint_argmax_all['std'].append(np.std(constraint_argmax))
            
        print ('---------------------------------------------------------')

    base_risk['mean'] = np.mean(base_risk_all)
    base_risk['std'] = np.std(base_risk_all)
    base_risk_argmax['mean'] = np.mean(base_risk_argmax_all)
    base_risk_argmax['std'] = np.std(base_risk_argmax_all)
            
    results = {'risk':risk_all,
               'constraint':constraint_all,
               'base_risk':base_risk,
               'risk_argmax':risk_argmax_all,
               'constraint_argmax':constraint_argmax_all,
               'base_risk_argmax':base_risk_argmax,
               'training_time_hist':time_hist}        
            
    return results

def evaluate_DP_unaware(dataset, num, K=2, T=5000, eps_list=[[0.1, 0.1], [0.01, 0.01]], 
                            print_details = True, beta='auto',
                            alg={'base':'SGD3', 'method':'ACSA'},
                            TRAIN_SIZE=0.4, UNLAB_SIZE=0.4, TEST_SIZE=0.2, data_scaling=False):
    #getting data
    if dataset=='adult':
        X, S, y = get_adult_data(problem='DP_unaware', as_df=False)
        #we take only 2000 samples for comparison
        sample_size = 2000 
        X, _, S, _, y ,_ = train_test_split(X, S, y, train_size = sample_size,  stratify=S, random_state=42)
        S_num = 2
        p = get_frequencies(S)
    elif dataset=='german':        
        X, S, y = get_german_data(problem='DP_unaware', as_df=False)
        S_num = 2
        p = get_frequencies(S)
    elif dataset=='communities':        
        X, S, y = get_communities_data(problem='DP_unaware', as_df=False)
        S_num = 2
        p = get_frequencies(S)
    else:
        raise Exception('Dataset not found.')

    #scaling
    if data_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    #initialization
    time_hist = []
    risk_all, risk_argmax_all = {'mean':[], 'std':[]}, {'mean':[], 'std':[]}
    base_risk, base_risk_all = {}, []
    base_argmax_risk, base_argmax_risk_all = {}, []
    unf_all, base_unf, base_unf_all = {}, {}, {}
    unf_argmax_all, base_unf_argmax, base_unf_argmax_all = {}, {}, {}
    for s in range(S_num):
        unf_all[s] = {'mean':[], 'std':[]}
        base_unf[s], base_unf_all[s] = {}, []
        unf_argmax_all[s] = {'mean':[], 'std':[]}
        base_unf_argmax[s], base_unf_argmax_all[s] = {}, []

    for k, eps in enumerate(eps_list):
        if print_details:
            print (k+1,'/',len(eps_list), ' : collecting statistics for eps='+str(eps))
        
        risk, risk_argmax = [], []
        unf = {}
        for s in range(S_num):
            unf[s] = []
    
        for i in range(1, num+1):

            X_, X_test, S_, S_test, y_, y_test = train_test_split(X, S, y, test_size=TEST_SIZE, stratify=S, random_state=i)
            X_train, X_unlab, S_train, S_unlab, y_train, y_unlab = train_test_split(X_, S_, y_, train_size = TRAIN_SIZE/(1-TEST_SIZE), 
                                                                stratify=S_,random_state=i)
            
            start = time.time()
            #training base classifier
            #base_clf = RandomForestClassifier(random_state=42)
            base_clf = LogisticRegression(multi_class="multinomial", solver='lbfgs', 
                                           max_iter=5000, random_state=i)
            base_clf.fit(X_train,y_train)

            #training tau classifier (for predicting S)
            tau_clf = LogisticRegression(random_state=42)
            tau_clf.fit(X_train,S_train)

            #training PostProc
            PostProc_clf = PostProc(problem='DP_unaware', base_classifier=base_clf, K=K, T=T,
                                    DPunaware_params = {'sens_num':S_num, 'sens_clf':tau_clf, 
                                                        'sens_freq':p, 'B':eps})
            
            PostProc_clf.fit(X=X_unlab, beta =beta, alg=alg)
            end = time.time()
            time_hist.append(end-start)
            
            pred_base = base_clf.predict_proba(X_test)
            pred_base_argmax = base_clf.predict(X_test)
            pred = PostProc_clf.predict(X_test)
            pred_argmax = np.argmax(pred, axis=1)

            risk.append(eval_risk(pred, y_test, 'DP_unaware'))    
            risk_argmax.append(eval_risk_clf(pred_argmax, y_test))          
            base_risk_all.append(eval_risk(pred_base, y_test, 'base'))
            base_argmax_risk_all.append(eval_risk_clf(pred_base_argmax, y_test))
            unf_DP = eval_DP(pred, S_test)
            base_unf_DP = eval_DP(pred_base, S_test)
            for s in range(S_num):
                unf[s].append(unf_DP[s])
                base_unf_all[s].append(base_unf_DP[s])

            if print_details:    
                print ('-----   ', i,'/',num,': training completed, statistics collected')
        
        risk_all['mean'].append(np.mean(risk))
        risk_all['std'].append(np.std(risk))
        for s in range(S_num):
            unf_all[s]['mean'].append(np.mean(unf[s]))
            unf_all[s]['std'].append(np.std(unf[s]))

        risk_argmax_all['mean'].append(np.mean(risk_argmax))
        risk_argmax_all['std'].append(np.std(risk_argmax))
            
        print ('---------------------------------------------------------')

    base_risk['mean'] = np.mean(base_risk_all)
    base_risk['std'] = np.std(base_risk_all)
    for s in range(S_num):
        base_unf[s]['mean'] = np.mean(base_unf_all[s])
        base_unf[s]['std'] = np.std(base_unf_all[s])

    base_argmax_risk['mean'] = np.mean(base_argmax_risk_all)
    base_argmax_risk['std'] = np.std(base_argmax_risk_all)
            
    results = {'risk':risk_all,
               'risk_argmax':risk_argmax_all,
               'unfairness':unf_all,
               'base_risk':base_risk,
               'base_risk_argmax':base_argmax_risk,
               'base_unfairness':base_unf,
               'training_time_hist':time_hist}        
            
    return results