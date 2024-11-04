import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from data_prep import get_adult_data, get_german_data
from models_for_comparison import BinClassRO
from PostProc import PostProc
from evaluation_measures import eval_risk, eval_risk_clf, eval_RejectOption, eval_RejectOption_clf, eval_DP

    
def evaluate_reject_option(dataset, num, K=2, T=5000, alpha_list=[0.1, 0.05, 0.025], 
                            print_details = True, beta='auto', L='auto',
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
            base_clf = LogisticRegression(random_state=i)
            base_clf.fit(X_train,y_train)

            #training PostProc
            start = time.time()
            PostProc_clf = PostProc(problem='reject_option', base_classifier=base_clf, K=K, T=T,
                                    RejectOption_params = {'B':alpha})
            PostProc_clf.fit(X=X_unlab, beta ='auto', alg=alg)
            end = time.time()
            time_hist.append(end-start)
            
            pred_base = base_clf.predict_proba(X_test)
            pred_argmax_base = base_clf.predict(X_test)
            pred = PostProc_clf.predict(X_test)
            pred_argmax = np.array(PostProc_clf.A)[np.argmax(pred, axis=1)]
            
            # #inverse scaling for evaluation
            # if y_scaling:
            #     y_test = scaler.inverse_transform(y_test.values.reshape(-1, 1))[:,0]

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