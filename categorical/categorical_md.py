import warnings
import tqdm as tqdm
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score,
                             precision_recall_curve)

import pymc3 as pm
import theano.tensor as tt
#from mlxtend.plotting import plot_confusion_matrix
import theano
from pymc3.variational.callbacks import CheckParametersConvergence
import statsmodels.formula.api as smf
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from IPython.display import HTML
from Dataset import SimulatedData
import time

SEED = 42
N_sample = 100
tune = 100
chains = 1
draws = 100
notice = 'pivoting_intercept'
model_path = "./simu_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')+ notice + '_' + str(N_sample) + '_' + str(draws) + '_' + str(tune) + '_' + str(chains) + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
np.random.seed(SEED)
dataset = SimulatedData(N_sample, outcome='ICU')
dataset.explore_dataset()
x = dataset['x'].values
y = dataset['y'].values
nbr_classes = dataset.nbr_classes
with pm.Model() as categorical_model:
    nbr_classes = nbr_classes
    a = pm.Normal("intercept", mu=0, sigma=100, shape=nbr_classes-1)  # intercepts
    b = pm.Normal("age", mu=0, sigma=10, shape=nbr_classes-1)
    c = pm.Normal("gender", mu=0, sigma=50, shape=nbr_classes-1)
    d = pm.Normal("smoking", mu=0, sigma=50, shape=nbr_classes-1)
    e = pm.Normal("fever", mu=0, sigma=50, shape=nbr_classes-1)
    f = pm.Normal("vomiting", mu=0, sigma=100, shape=nbr_classes-1)
    s0 = a[0] + b[0] * x[:, 0] + c[0] * x[:, 1] + d[0] * x[:, 2] + e[0] * x[:, 3] + f[
        0] * x[:, 4]
    #s1 = a[1] + b[1] * x[:, 0] + c[1] * x[:, 1] + d[1] * x[:, 2] + e[1] * x[:, 3] + f[
        #1] * x[:, 4]
    #  s2 = a[2] + b[2] * x[:, 0] + c[2] * x[:, 1] + d[2] * x[:, 2] + e[2] * x[:, 3] + f[2] * x[:, 4]
    s2 = np.zeros(x[:, 0].shape[0])#  pivoting the intercept for the third category
    s = pm.math.stack([s2, s0]).T
    p_ = tt.nnet.softmax(s)
    outcome_obs = pm.Categorical("outcome", p=p_, observed=y)

    trace_multinomial = pm.sample(draws=draws,
                                  tune=tune,
                                  chains=chains,
                                  cores=1,
                                  init='auto', progressbar=True)

    print(categorical_model.basic_RVs)
    pm.plot_trace(trace_multinomial)
    plt.savefig(model_path + 'trace')
    pm.summary(trace_multinomial).to_csv(model_path + 'trace.csv')
    print(pm.summary(trace_multinomial))
    az.plot_pair(trace_multinomial)
    plt.savefig(model_path + 'plot_pair')
    pm.energyplot(trace_multinomial)
    plt.savefig(model_path + 'energy')
    az.plot_forest(trace_multinomial)
    plt.savefig(model_path + 'forest')
    pm.plot_posterior(trace_multinomial)
    plt.savefig(model_path + 'posterior')

