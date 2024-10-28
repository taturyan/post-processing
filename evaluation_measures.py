import numpy as np

def risk(y, pred, problem='reject'):
    """
    Args:
        base_pred (array): probabilistic predictions predictied by base_classifier
        pred (array): probabilistic predictions predictied after post-processing
        problem (str): 'reject_option', 'alpha_risk', 'DP_unaware', 'DP_aware', 'general'

    Returns:
        int: the risk /used for evaluation/
    """
    if problem=='reject_option':
        pred = pred[:, :-1]

    return np.mean(np.sum((1 - np.eye(pred.shape[1])[y])*pred, axis=1))

def prob_risk(base_pred, pred, problem='reject'):
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

def gen_risk(ell, pred):
    pass

############### constraints evaluation ###############

def eval_RejectOption(pred):
    return np.mean(pred[:,-1])

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

def eval_gen_constraint(c):
    pass