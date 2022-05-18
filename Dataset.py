import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
import tensorflow_probability as tfp
import matplotlib.cm as cm

class SimulatedData:

    def __init__(self, n_sample, outcome='severity level', missing=None):
        self.path = '../data/'
        self.type = outcome
        self.N = n_sample
        self.prob = np.reshape(np.array([[0.50, 0.50], [0.3, 0.3, 0.3, 0.1], [0.2, 0.8], [0.7, 0.3]]), (4, -1))
        self.age = np.random.uniform(35/70, 70/70, n_sample)
        self.gender = np.random.choice([0, 1], n_sample, p=self.prob[0, 0])
        self.smoking = np.random.choice([0, 5, 6, 7], n_sample, p=self.prob[1, 0])# [0, 5, 6, 7]
        self.fever = np.random.choice([0, 1], n_sample, p=self.prob[2, 0])
        self.vomiting = np.random.choice([0, 1], n_sample, p=self.prob[3, 0])
        # possible other variable
        #  self.race = np.random.choice([0, 1], N_sample, p=[0.5, 0.5])
        #  self.bmi = np.random.uniform(5, 50, N_sample).astype(int)
        self.coeff = [1.25*70, 22, 8, 34, -45]
        if missing is not None:
            idx_missing = np.random.uniform(0, int(n_sample*missing-1), int(n_sample*missing)).astype(int)
            self.age[idx_missing] = 8
        self.score = np.zeros(self.N)
        if self.type == 'severity level':
            self.prob_vec = np.zeros((self.N, 3))
            self.outcome = self.generate_ordered_outcome()
        elif self.type == 'ICU':
            self.prob_vec = np.zeros((self.N, 2))
            self.outcome = self.generate_binary_outcome()
        self.nbr_classes = np.unique(self.outcome).shape[0]
        self.data = list(zip(self.age, self.gender, self.smoking, self.fever, self.vomiting, self.outcome))
        df = pd.DataFrame(self.data,
                          columns=['age', 'gender', 'smoking', 'fever', 'vomiting', 'outcome'])
        df.to_csv(self.path+'data.csv', index=False)
        self.data = pd.read_csv(self.path+'data.csv')
        self.data.info()
        self.data.head()

    def get_true_coeff(self):
        if self.type == 'severity level':
            self.coeff.append(self.A)
            self.coeff.append(self.B)
        elif self.type == 'ICU':
            self.coeff.insert(0, -self.tresh)
        return self.coeff

    def generate_binary_outcome(self):
        score = np.zeros_like(self.age)
        outcome = np.zeros_like(self.age)
        for i in range(self.age.shape[0]):
            score[i] += self.coeff[0] * self.age[i]
            score[i] += self.coeff[1] * self.gender[i]
            score[i] += self.coeff[2] * self.smoking[i]
            score[i] += self.coeff[3] * self.fever[i]
            score[i] += self.coeff[4] * self.vomiting[i]
        self.tresh = np.percentile(score, 50)
        for i in range(score.shape[0]):
            prob = np.array(tfp.distributions.OrderedLogistic(cutpoints=[float(self.tresh)],
                                                              loc=float(score[i])).categorical_probs())
            self.score[i] = score[i]
            self.prob_vec[i, :] = prob
            outcome[i] = np.random.choice([0, 1], 1, p=prob)
        return outcome

    def generate_ordered_outcome(self):
        score = np.zeros_like(self.age)
        outcome = np.zeros_like(self.age)
        for i in range(self.age.shape[0]):
            score[i] += self.coeff[0] * self.age[i]
            score[i] += self.coeff[1] * self.gender[i]
            score[i] += self.coeff[2] * self.smoking[i]
            score[i] += self.coeff[3] * self.fever[i]
            score[i] += self.coeff[4] * self.vomiting[i]

        self.A = np.percentile(score, 33)
        self.B = np.percentile(score, 66)

        for i in range(score.shape[0]):
            prob = np.array(tfp.distributions.OrderedLogistic(cutpoints=[float(self.A), float(self.B)],
                                                              loc=float(score[i])).categorical_probs())
            self.score[i] = score[i]
            self.prob_vec[i, :] = prob

            outcome[i] = np.random.choice([0, 1, 2], 1, p=prob)

            '''if score[i] <= self.A:
                outcome[i] = 0
            elif score[i] >= self.B:
                outcome[i] = 2
            else:
                outcome[i] = 1'''
        return outcome

    def explore_dataset(self):

        data = self.data
        if self.type == 'severity level':
            plt.figure()
            idx = np.where(self.outcome == 0)
            plt.scatter(self.outcome[idx], self.prob_vec[idx, 0])
            idx = np.where(self.outcome == 1)
            plt.scatter(self.outcome[idx], self.prob_vec[idx, 1])
            idx = np.where(self.outcome == 2)
            plt.scatter(self.outcome[idx], self.prob_vec[idx, 2])
            plt.savefig(self.path + 'oredered_prob_related.png')
            plt.figure()
            idx = np.argsort(self.score)
            plt.plot(self.score[idx], self.prob_vec[idx, 0])
            plt.plot(self.score[idx], self.prob_vec[idx, 2])
            plt.plot(self.score[idx], self.prob_vec[idx, 1])
            plt.plot([self.A, self.A], [-0.2, 1.2], color='black')
            plt.plot([self.B, self.B], [-0.2, 1.2], color='black')
            plt.ylim([-0.2, 1.2])
            plt.xlim(self.A-20, self.B+20)
            plt.savefig(self.path + 'ordered_data_class_prob.png')

        elif self.type == 'ICU':
            plt.figure()
            idx = np.where(self.outcome == 0)
            plt.scatter(self.outcome[idx], self.prob_vec[idx, 0])
            idx = np.where(self.outcome == 1)
            plt.scatter(self.outcome[idx], self.prob_vec[idx, 1])
            plt.savefig(self.path + 'binary_prob_related.png')
            plt.figure()
            idx = np.argsort(self.score)
            plt.plot(self.score[idx], self.prob_vec[idx, 0])
            plt.plot(self.score[idx], self.prob_vec[idx, 1])
            plt.plot([-0.2, 1.2], [self.tresh, self.tresh], color='black')
            plt.ylim([-0.2, 1.2])
            # plt.xlim(self.tresh - 20, self.tresh + 20)
            plt.savefig(self.path + 'binary_data_class_prob.png')

        # plotpair
        plt.figure()
        seaborn.pairplot(data)
        plt.title('Data pairs distribution')
        plt.savefig(self.path + 'data_pairplot.png')

        #  plot correlation
        '''# Compute the correlation matrix
        corr = data.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = seaborn.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        seaborn.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=0.3,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            ax=ax,
        )'''
        plt.figure(figsize=(12, 8))
        corr = data.corr()
        mask = np.tri(*corr.shape).T
        seaborn.heatmap(corr.abs(), mask=mask, annot=True)
        b, t = plt.ylim()
        b += 0.5
        t -= 0.5
        plt.ylim(b, t)
        plt.title('Correlation between data')
        plt.savefig(self.path + 'data_correlation.png')

        # see impact to the target
        plt.figure()
        n_fts = len(data.columns)
        colors = cm.rainbow(np.linspace(0, 1, n_fts))
        data.drop('outcome', axis=1).corrwith(data.outcome).sort_values(ascending=True).plot(kind='barh',
                                                                                             color=colors,
                                                                                             figsize=(12, 8))
        plt.title('Correlation to Target (outcome)')
        plt.savefig(self.path + 'data_corr2target.png')
        plt.figure()

    def __getitem__(self, item):
        if item == 'x':
            return self.data.drop('outcome', axis=1)
        elif item == 'y':
            return self.data.outcome
        else:
            print('error')
