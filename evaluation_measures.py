import numpy as np

############### risk evaluation measures ###############

def eval_risk(y_pred_prob, y_true, problem='reject_option'):
    """
    Args:
        pred (array): probabilistic predictions predictied after post-processing
        pred (array): true labels
        problem (str): 'reject_option', 'alpha_risk', 'DP_unaware', 'DP_aware', 'general', 'base'
    Returns:
        int: the risk /used for evaluation/
    """
    if problem=='reject_option':
        y_pred_prob = y_pred_prob[:, :-1]
    return np.mean(np.sum((1 - np.eye(y_pred_prob.shape[1])[y_true])*y_pred_prob, axis=1))

def eval_risk_clf(y_pred, y_true):
    if 'R' in y_pred.astype(str):
        y_pred_ = y_pred[y_pred!='R']
        y_pred_ = y_pred_.astype(int)
        y_true = y_true[y_pred!='R']
    else:
        y_pred_ = y_pred.astype(int)
    return np.sum(y_pred_!=y_true)/len(y_pred)

def eval_optim_risk(base_pred, pred, problem='reject_option'):
    """
    Args:
        base_pred (array): probabilistic predictions predictied by base_classifier
        pred (array): probabilistic predictions predictied after post-processing
        problem (str): 'reject_option', 'alpha_risk', 'DP_unaware', 'DP_aware', 'general'
    Returns:
        int: the minimized risk /NOT USED FOR EVALUATION/
    """
    if problem=='reject_option':
        return np.mean(np.sum((1-base_pred)*pred[:, :-1], axis=1))
    else:
        return np.mean(np.sum((1-base_pred)*pred, axis=1))

def eval_gen_risk(ell, pred):
    pass

############### constraints evaluation measures ###############

def eval_RejectOption(pred_prob):
    return np.mean(pred_prob[:,-1])

def eval_RejectOption_clf(y_pred):
    if 'R' in y_pred.astype(str):
        return len(y_pred[y_pred=='R'])/len(y_pred)
    else:
        return 0

def eval_AlphaRisk(f_pred, pred):
    return np.mean(np.sum((1 - np.eye(pred.shape[1])[f_pred])*pred, axis=1))

def eval_DP(pred, S):
    PMF = pred.mean(axis=0)
    CDF=np.cumsum(PMF)
    
    Unfairness = {}
    S_val = sorted(S.unique())
    for s in S_val:
        PMF_s = pred[S==s].mean(axis=0)
        CDF_s = np.cumsum(PMF_s)
        Unfairness[s]=max(abs(CDF_s-CDF))
    return Unfairness    

def eval_DP_argmax(pred_argmax, S):
    bins=len(np.unique(pred_argmax))
    hist, bin_edges = np.histogram(pred_argmax,bins=bins)
    CDF = np.cumsum(hist/len(pred_argmax))
    Unfairness = {}
    S_val = sorted(S.unique())
    for s in S_val:
        hist, bin_edges = np.histogram(pred_argmax[S==s],bins=bins)
        CDF_s = np.cumsum(hist/sum(hist))
        Unfairness[s]=max(abs(CDF_s-CDF))
    return Unfairness

def eval_gen_constraint(c):
    pass