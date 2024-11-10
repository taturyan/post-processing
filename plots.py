import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#plot different types of unfairness vs risk wrt pairs of epsilon thresholds      
def plot_compare(pairs_list, model_list,
                markers_list=['o','s','x'], dataset='adult', 
                x_label = 'constraint', y_label = 'risk',
                colors = ['tab:green','tab:orange'], legend_size=8, alpha=0.7, 
                plot_std=True, annotate=False, loglog=False, start_0=False, linestyle='dashed', 
                figsize=(14,5), fontsize=10):
    
    plt.figure(figsize=figsize, dpi=200)
    plt.title(dataset, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    
    for i, pair in enumerate(pairs_list):
        constraint, risk = pair[0], pair[1]

        LABEL = str(model_list[i])
            
        if 'base' in model_list[i]:
            ALPHA=1
        else:
            ALPHA=alpha
                
        cnstr_=constraint['mean']
        cnstr_std=constraint['std']

        risk_=risk['mean']
        risk_std=risk['std']
                   
        line, = plt.plot(cnstr_, risk_, label=LABEL, marker=markers_list[i], linestyle=linestyle, color=colors[i])
        line.set_alpha(ALPHA)
        if plot_std:
            _, caps, bars = plt.errorbar(cnstr_, risk_, yerr=risk_std, xerr=cnstr_std, color=colors[i], 
                                         linestyle='None')
            for bar in bars:
                bar.set_alpha(0.15)
                
        if annotate:
            if 'base' not in model_list[i]:
                for j in range(len(risk_)):
                    plt.annotate('B'+str(j+1), (cnstr_[j], risk_[j]), fontsize=8)
            
    plt.legend(prop={'size': legend_size})
    if start_0:
        plt.xlim(left=0)
        plt.xlim(right=0.7)
        plt.ylim(bottom=0)
        plt.ylim(top=0.1)
    plt.show()