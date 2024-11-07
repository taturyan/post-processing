import numpy as np
from scipy.special import softmax
import collections
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

#from evaluation_measures import DP_unfairness, prob_unfairness, unfairness
class PostProc:
    def __init__(self, problem, base_classifier, K, T,
                 RejectOption_params = {'B':None},
                 AlphaRisk_params = {'f_clf': None, 'B':None},
                 DPunaware_params = {'sens_num':None, 'sens_clf':None, 'sens_freq':None, 'B':None},
                 general_params = {'A':None, 'ell': None, 'c': None, 'B': None},
                 keep_history=False):
        """
        Args:
            problem (str): 'reject_option', 'alpha_risk', 'DP_unaware', 'DP_aware', 'general'
            base_classifier (function): base classifier that returns probabilistic predictions with predict_proba method
            K (int): number of classes
            T (int): number of iterations
            RejectOption_params (dict, optional): parameters for problem = 'reject_option'
                B (int) - constraint violation threshold
            AlphaRisk_params (dict, optional): parameters for problem = 'alpha_risk'
                f_clf (function) - predicts single class given X; 
                B (int) - - constraint violation threshold
            DPunaware_params (dict, optional): parameters for problem = 'DP_unaware'
                sens_num (int) - number of sensitive attributes
                sens_clf (function) - classifier that returns probabilistic predictions S|X with predict_proba method 
                frequencies (list) - list of marginal distribution of sensitive attribute, 
                B (list) - list of constraint violation thresholds [eps_1,...,eps_sens_num]
            general_params (dict, optional): parameters for problem = 'general'
                A (list) - list of possible predictions;
                ell (function) - loss function ell(x,a);
                c (list) - list of constraint functions c_j(x,a) for j=1,...,M;
                B (list) - list of constraint violation thresholds [B_1,...,B_M]
            keep_history (bool, optional): if true, estimators are saved at each iteration. Defaults to False.
        """
        self.problem = problem
        self.base_classifier = base_classifier 
        self.K = K 
        self.T = T 
        self.RejectOption_params = RejectOption_params 
        self.AlphaRisk_params = AlphaRisk_params 
        self.DPunaware_params = DPunaware_params 
        self.general_params = general_params 
        self.keep_history = keep_history 
        
    def set_problem(self):
        if self.problem == 'reject_option':
            self.A = [k for k in range(self.K)]
            self.A.append('R')
            self.B = self.RejectOption_params['B']
            self.stoch_grad = self.stoch_grad_RejectOption
            self.w_0 = np.zeros(1) #initial point
            self.sigma_sq = 1 #variance
            self.predict = self.predict_RejectOption
        elif self.problem == 'alpha_risk':
            self.A = [k for k in range(self.K)]
            self.f_clf = self.AlphaRisk_params['f_clf']
            self.B = self.AlphaRisk_params['B']
            self.stoch_grad = self.stoch_grad_AlphaRisk
            self.w_0 = np.zeros(1) #initial point
            self.sigma_sq = self.K-1 #variance
            self.predict = self.predict_AlphaRisk
        elif self.problem == 'DP_unaware':
            self.A = [k for k in range(self.K)]
            self.B = self.DPunaware_params['B']
            self.tau_X = self.DPunaware_params['sens_clf']
            self.S_num = self.DPunaware_params['sens_num']
            self.p = self.DPunaware_params['sens_freq']
            self.sum_ps=0
            for p_s in self.p:
                self.sum_ps += (1-p_s)/p_s
            self.sigma_sq = 2*self.sum_ps #variance
            self.stoch_grad = self.stoch_grad_DPunaware
            self.w_0 = np.zeros(2*self.K*self.S_num) #initial point
            self.predict = self.predict_DPunaware
        elif self.problem == 'DP_aware':
            self.A = [k for k in range(self.K)]
            self.stoch_grad = self.stoch_grad_DPaware
            self.w_0 = np.zeros(2*self.K*self.S_num) #initial point
            self.sigma_sq = None #TODO: compute
            self.predict = self.predict_DPaware
        elif self.problem == 'general':
            pass
        else:
            raise Exception('Problem not found.')
            
    ####### stochastic gradient functions for each example #######
                  
    def stoch_grad_RejectOption(self, w, x):
        clf_prob = self.base_classifier.predict_proba(x.reshape(1, -1))
        grad =  - np.exp(-w) / (np.sum(np.exp(clf_prob-1)) + np.exp(-w)) + self.B
        return grad
    
    def stoch_grad_AlphaRisk(self, w, x):
        clf_prob = self.base_classifier.predict_proba(x.reshape(1, -1))[0]
        vec = clf_prob - 1 - w*np.array([int(self.f_clf.predict(x.reshape(1, -1))!=k) for k in range(self.K)])
        grad = - np.sum(softmax(self.beta*vec)) + self.B
        return grad
    
    def stoch_grad_DPunaware(self, w, x):
        clf_prob = self.base_classifier.predict_proba(x.reshape(1, -1))[0]
        tau_x_prob = self.tau_X.predict_proba(x.reshape(1, -1))
        tau_x_coef = np.zeros(tau_x_prob.shape)
        for i, p_i in enumerate(self.p):
            tau_x_coef[:,i] = 1 - tau_x_prob[:,i]/p_i 

        grad = np.zeros(2*self.K*self.S_num)
        diff = (w[:self.S_num*self.K].copy() - w[self.S_num*self.K:].copy()).reshape((self.S_num,self.K))
        softmax_x = softmax(self.beta*(clf_prob - 1 + np.matmul(tau_x_coef,diff)), axis = 1)
        for i in range(self.S_num):
            grad[i*self.K:(i+1)*self.K] = np.mean(tau_x_coef[:,i][:, None]*softmax_x,axis=0) + self.B[i]
            grad[(i+self.S_num)*self.K:(i+1+self.S_num)*self.K] = -np.mean(tau_x_coef[:,i][:, None]*softmax_x,axis=0) + self.B[i]
        return grad
    
    def stoch_grad_DPaware(self, w, x):
        pass

    def stoch_grad_general(self, w, x):
        pass

    
    ####### regularization of stochastic gradient #######
    
    def stoch_grad_reg(self, w, x, w_reg=[], mu_reg=[]):
        reg = 0
        for i in range(len(w_reg)):
            reg += mu_reg[i]*(w-w_reg[i])
        return self.stoch_grad(w, x) + reg
    
    ####### optimization algorithms #######

    def SGD(self, X, w_init, alpha, T, w_reg=[], mu_reg=[]):
        w = w_init
        w_all = []
        w_hist = []
        for t in range(T):
            x = X[np.random.randint(len(X))]  
            w -= alpha * self.stoch_grad_reg(w, x, w_reg, mu_reg)
            w[w<0] = 0
            w_all.append(w)
            if self.keep_history:
                w_hist.append(np.mean(w_all, axis=0))

        return np.mean(w_all, axis=0), w_hist  
    
    def SGD_sc(self, X, w_init, mu, L, T, w_reg=[], mu_reg=[]):
        N1 = int(np.floor(T / (L / mu)))
        K1 = int(np.floor(np.log2(mu * T / L)))
        w = w_init
        w_hist = []
        for t in range(0, N1):
            w, w_hist_ = self.SGD(X, w, 1 / (2 * L), int(np.floor(L / mu)), w_reg, mu_reg)
            if self.keep_history:
                w_hist+=w_hist_
        for k in range(0, K1):
            w, w_hist_ = self.SGD(X, w, 1 / (2**k * L), int(np.floor((2**(k + 2)) * L / mu)), w_reg, mu_reg)
            if self.keep_history:
                w_hist+=w_hist_
        return w, w_hist
    
    def AC_SA(self, X, w_init, mu, L, T, w_reg=[], mu_reg=[]):
        w = w_init
        w_ag = w_init
        w_hist = []
        
        for i in range(T):
            alpha = 2 / (i+1)
            gamma = 4 * L / ((i+1) * (i+2))
            alpha1 = 1 - alpha

            coef1 = (alpha1 * (mu + gamma)) / (gamma + (1 - alpha ** 2) * mu)
            coef2 = alpha * (alpha1 * mu + gamma) / (gamma + (1 - alpha ** 2) * mu)
            coef3 = alpha * mu / (mu + gamma)
            coef4 = (alpha1 * mu + gamma) / (mu + gamma)
            coef5 = alpha / (mu + gamma)

            w_md = coef1 * w_ag + coef2 * w
            x = X[np.random.randint(len(X))]
            w = coef3 * w_md + coef4 * w - coef5 * self.stoch_grad_reg(w_md, x, w_reg, mu_reg)
            w[w<0] = 0
            w_ag = alpha * w + alpha1 * w_ag
            
            if self.keep_history:
                w_hist.append(w_ag)
                
        return w_ag, w_hist
    
    
    def AC_SA_2(self, X, w_init, mu, L, T, w_reg=[], mu_reg=[]):
        
        w_hist = []

        w_ag_1, w_hist_ = self.AC_SA(X, w_init, mu, L, T, w_reg, mu_reg)
        if self.keep_history:
            w_hist+=w_hist_
        w_ag_2, w_hist_ = self.AC_SA(X, w_ag_1, mu, L, T, w_reg, mu_reg)
        if self.keep_history:
            w_hist+=w_hist_
            
        return w_ag_2, w_hist
        

    def SGD3_sc(self, X, w_init, mu_init, L, T, method, sc = True):
        w = w_init
        new_init = w_init
        mu = mu_init
        w_reg = []
        mu_reg = []
        w_history = []

        if sc == False:
            w_reg.append(w)
            mu_reg.append(w)
        S1 = int(np.floor(np.log2(L/mu)))
        for s in range(1, S1+1):
            if method=='ACSA':
                w, w_hist = self.AC_SA(X, new_init, mu, L,
                                                  T=int(np.floor(T/S1)), w_reg=w_reg, mu_reg=mu_reg)
            elif method=='ACSA2':
                w, w_hist = self.AC_SA_2(X, new_init, mu, L,
                                                  T=int(np.floor(T/S1)), w_reg=w_reg, mu_reg=mu_reg)
            elif method=='sgd':
                w, w_hist = self.SGD_sc(X, w, mu, 3*L, int(np.floor(T/S1)), w_reg, mu_reg)
            else:
                raise Exception('Method not found.')
            new_init = w_init
            mu = 2*mu
            w_reg.append(w)
            mu_reg.append(mu)
            if self.keep_history:
                w_history+=w_hist
        return w, w_history

    def SGD3(self, X, w_init, mu, L, T, method='ACSA2'):
        return self.SGD3_sc(X, w_init, mu, L + mu, T, method, sc = False)
    
    ####### fitting #######

    def fit(self, X, beta = 'auto', alg={'base':'SGD3', 'method':'ACSA2'}):
        
        self.N = len(X)
        
        if beta == 'auto':
            self.beta = 0.5 * np.sqrt(self.N) * np.log2(self.N) #temperature param in softmax
        else:
            self.beta = beta
        
        self.set_problem() #setting example-specific functions and parameters

        self.L = self.beta*self.sigma_sq #Lipschits constant
        self.mu =  self.L / self.T
        
        #self.stoch_grad_counter = 0
        
        if alg['base']=='SGD3':
            if len(alg)>0:
                self.w_est, self.w_est_hist = self.SGD3(X, self.w_0, self.mu, self.L, self.T, method=alg['method'])
            else:
                self.w_est, self.w_est_hist = self.SGD3(X, self.w_0, self.mu, self.L, self.T, method='ACSA2')
        elif alg['base']=='SGD':
            self.w_est, self.w_est_hist = self.SGD(X, self.w_0, 1/self.L, self.T)
        elif alg['base']=='ACSA':
            self.w_est, self.w_est_hist = self.AC_SA(X, self.w_0, self.mu, self.L, self.T)
        elif alg['base']=='ACSA2':
            self.w_est, self.w_est_hist = self.AC_SA_2(X, self.w_0, self.mu, self.L, self.T)
        else:
            raise Exception('Method not found.')            
            
    ####### prediction functions for each example #######
    
    def predict_RejectOption(self, X, lmbd = 'optimal'):
        if lmbd == 'optimal':
            lmbd = self.w_est
        else:
            lmbd = lmbd 

        clf_prob = self.base_classifier.predict_proba(X)
       
        vec = np.empty((clf_prob.shape[0], clf_prob.shape[1]+1)) 
        vec[:, :clf_prob.shape[1]] = clf_prob - 1
        vec[:, clf_prob.shape[1]] = -lmbd
        
        self.pred_prob = softmax(self.beta*vec, axis = 1)
        
        return self.pred_prob
    
        
    def predict_AlphaRisk(self, X, lmbd = 'optimal'):
        if lmbd == 'optimal':
            lmbd = self.w_est
        else:
            lmbd = lmbd 

        clf_prob = self.base_classifier.predict_proba(X)
        f_X = self.f_clf.predict(X)
        vec = clf_prob - 1 - lmbd*(1 - np.eye(self.K)[f_X])
        self.pred_prob = softmax(self.beta*vec, axis = 1)

        return self.pred_prob
    
    def predict_DPunaware(self, X, lmbd = 'optimal'):
        if lmbd == 'optimal':
            lmbd = self.w_est
        else:
            lmbd = lmbd 

        clf_prob = self.base_classifier.predict_proba(X)
        tau_X_prob = self.tau_X.predict_proba(X)
        tau_X_coef = np.zeros(tau_X_prob.shape)
        for i, p_i in enumerate(self.p):
            tau_X_coef[:,i] = 1 - tau_X_prob[:,i]/p_i 

        diff = (lmbd[:self.S_num*self.K].copy() - lmbd[self.S_num*self.K:].copy()).reshape((self.S_num,self.K))
        pred_prob = softmax(self.beta*(clf_prob - 1 + np.matmul(tau_X_coef,diff)), axis = 1)
        self.pred_prob = pred_prob
        
        return self.pred_prob
                          
    def predict_DPaware(self, X, lmbd = 'optimal'):
        if lmbd == 'optimal':
            lmbd = self.w_est
        else:
            lmbd = lmbd 

        pass

    def predict_general(self, X, lmbd = 'optimal'):
        if lmbd == 'optimal':
            lmbd = self.w_est
        else:
            lmbd = lmbd 

        pass
    
    
    ####### getting history ####### 
    #TODO: modify to make simpler according to new prediction functions
    
    
#     def get_training_history(self, X, S, y, data_scaling=False, scaler=None):
        
#         reg_pred = self.base_method.predict(X)
#         clf_prob = self.classifier.predict_proba(X)
        
#         tau_X_coef = np.zeros(clf_prob.shape)
#         for i, p_i in enumerate(self.p):
#             tau_X_coef[:,i] = 1 - clf_prob[:,i]/p_i 
            
#         r_X = np.square(reg_pred[:, np.newaxis] - self.Q_L)
#         #inverse scaling for evaluation
#         if data_scaling:
#             y = scaler.inverse_transform(y.values.reshape(-1, 1))[:,0]
#             grid = scaler.inverse_transform(self.Q_L.reshape(-1, 1))[:,0] 
#         else:
#             grid = self.Q_L

#         r_X_y = np.square(y[:, np.newaxis] - grid)
        
#         risk_history = []
#         unfairness_history = []
        
#         for w_est in self.w_est_hist:
#             diff = (w_est[:self.K*(2*self.L+1)].copy() - w_est[self.K*(2*self.L+1):].copy()).reshape((self.K,2*self.L+1))
#             pred_prob = softmax(self.beta*(np.matmul(tau_X_coef,diff) - r_X), axis = 1)
            
#             risk_history.append(np.mean(np.sum(r_X_y*pred_prob, axis=1))) #probabilistic risk history
#             unfairness_history.append(unfairness(pred_prob, S)) #unfairness history
            
#         return risk_history, unfairness_history
        
        

