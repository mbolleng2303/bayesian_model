import pickle
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
from mlxtend.plotting import plot_confusion_matrix
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
from utils import make_plot
SEED = 42
N_sample = 100
tune = 500
chains = 2
draws = 500
notice = 'no_missing'
model_path = "./result/binomial_" + time.strftime('%Hh%Mm_%m_%d_%y') + '_' + notice + '_' + str(N_sample) + '_' + str(draws) + '_' + str(tune) + '_' + str(chains) + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
print(model_path)
np.random.seed(SEED)
dataset = SimulatedData(N_sample, outcome='ICU')
dataset.explore_dataset()
x = dataset['x'].values
y = dataset['y'].values
true_coeff = dataset.get_true_coeff()
nbr_classes = dataset.nbr_classes
labels = dataset['x'].columns
with pm.Model() as binomial_model:
    # pm.glm.GLM(x=x, labels=labels, y=y, family=pm.glm.families.Binomial())
    '''a = pm.Normal("intercept", mu=true_coeff[0], sigma=5)  # intercepts
    b = pm.Normal("age", mu=true_coeff[1], sigma=5)
    c = pm.Normal("gender", mu=true_coeff[2], sigma=5)
    d = pm.Normal("smoking", mu=true_coeff[3], sigma=5)
    e = pm.Normal("fever", mu=true_coeff[4], sigma=5)
    f = pm.Normal("vomiting", mu=true_coeff[5], sigma=5)'''
    a = pm.Normal("intercept", mu=0, sigma=100)  # intercepts
    b = pm.Normal("age", mu=0, sigma=10)
    c = pm.Normal("gender", mu=0, sigma=50)
    d = pm.Normal("smoking", mu=0, sigma=50)
    e = pm.Normal("fever", mu=0, sigma=50)
    f = pm.Normal("vomiting", mu=0, sigma=100)
    phi = a + b * x[:, 0] + c * x[:, 1] + d * x[:, 2] + e * x[:, 3] + f * x[:, 4]
    logit = pm.invlogit(phi)
    pm.Bernoulli(name='logit', p=logit, observed=y)
    trace_binomial = pm.sample(draws=draws,
                               tune=tune,
                               chains=chains,
                               cores=1,
                               init='auto', progressbar=True)
    with open(model_path + 'model.pkl', 'wb') as buff:
        pickle.dump({'model': binomial_model, 'trace': trace_binomial}, buff)
    with open(model_path + 'model.pkl', 'rb') as buff:
        data0 = pickle.load(buff)
    model, trace = data0['model'], data0['trace']
    make_plot(model, trace, true_coeff, model_path)

