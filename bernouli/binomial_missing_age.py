import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pymc3 as pm
from Dataset import SimulatedData
import time
from utils import make_plot

SEED = 42
N_sample = 100
tune = 500
chains = 2
draws = 500
notice = 'missing_age'
model_path = "./result/binomial_" + "_" + time.strftime('%Hh%Mm_%m_%d_%y') + '_' + notice + '_' + str(N_sample) + '_' + str(draws) + '_' + str(tune) + '_' + str(chains) + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
np.random.seed(SEED)
dataset = SimulatedData(N_sample, outcome='ICU') #  missing=0.2
dataset.explore_dataset()
x = dataset['x'].values
y = dataset['y'].values
true_coeff = dataset.get_true_coeff()
nbr_classes = dataset.nbr_classes
labels = dataset['x'].columns
# mask data
missing = 0.1
idx_missing = np.random.choice(x[:, 0].shape[0], int(N_sample*missing), replace=False)
previous = x[idx_missing, 0]
x[idx_missing, 0] = -1
idx_missing = np.where(x[:, 0] == -1)
idx_not_missing = np.where(x[:, 0] != -1)

with pm.Model() as binomial_model:
    v = pm.Uniform(name="age_estimation", lower=x[idx_not_missing, 0].min(), upper=x[idx_not_missing, 0].max(), shape=np.array(idx_missing).shape[1])
    #  pm.glm.GLM(x=x, labels=labels,y=y, family=pm.glm.families.Binomial())
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
    phi = a + b * x[idx_not_missing, 0] + c * x[idx_not_missing, 1] + d * x[idx_not_missing, 2] + e * x[idx_not_missing,3] + f * x[idx_not_missing, 4]
    phi_no = a + b * v[:] + c * x[idx_missing, 1] + d * x[idx_missing, 2] + e * x[idx_missing,3] + f * x[idx_missing, 4]
    like = pm.invlogit(phi)
    like_no = pm.invlogit(phi_no)
    pm.Bernoulli(name='logit', p=like, observed=y[idx_not_missing])
    pm.Bernoulli(name='logit_non', p=like_no, observed=y[[idx_missing]])

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
    make_plot(binomial_model, trace_binomial, true_coeff,
              model_path, var_names = ['intercept', 'age', 'gender', 'smoking', 'fever', 'vomiting'])
    make_plot(model, trace, previous,
              model_path + 'missing data/', var_names=['age_estimation'])



