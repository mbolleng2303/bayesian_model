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

dataset = SimulatedData(100, outcome='ICU')
model_path = '../backup/simu_15h37m14s_on_May_15_2022no_missing_100_100_100_1/'
true_coeff = dataset.get_true_coeff()
with open(model_path + 'model.pkl', 'rb') as buff:
    data0 = pickle.load(buff)
    model, trace = data0['model'], data0['trace']


def make_plot(model, trace, true_coeff, model_path) :
    print(model.basic_RVs)

    az.plot_trace(trace)
    plt.savefig(model_path + 'trace')

    az.summary(trace).to_csv(model_path + 'trace.csv')
    print(az.summary(trace))

    az.plot_pair(trace, kind=["scatter", "kde"], point_estimate="median")
    plt.savefig(model_path + 'plot_pair')

    az.plot_energy(trace)
    plt.savefig(model_path + 'energy')

    az.plot_forest(trace)
    for j, (y_tick, frac_j) in enumerate(zip(plt.gca().get_yticks(), reversed(true_coeff))):
        plt.vlines(frac_j, ymin=y_tick - 0.45, ymax=y_tick + 0.45, color="black", linestyle="--")
    plt.savefig(model_path + 'forest')

    az.plot_posterior(trace, ref_val=true_coeff)
    plt.savefig(model_path + 'posterior')

make_plot(model, trace, true_coeff, model_path)
