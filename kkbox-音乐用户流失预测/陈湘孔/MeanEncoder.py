import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
from itertools import product


class MeanEncoder:

    def __init__(self, features, n_splits=5, target_type='classification', k=2.0, f=1.0):
        """
        :param features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param k:1 / (1 + np.exp((n - k) / f))
        :param f:控制函数在拐点附近的斜率，f越大，“坡”越缓。
        """

        self.features = features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        self.prior_weight_func = lambda n: 1 / (1 + np.exp((n - k) / f))

    @staticmethod
    def mean_encode_subroutine(x_train, y_train, x_test, variable, target, prior_weight_func):
        x_train = x_train[[variable]].copy()
        x_test = x_test[[variable]].copy()

        if target is None:
            nf_name = '{}_predict'.format(variable)
            x_train['predict_temp'] = y_train  # regression
        else:
            nf_name = '{}_predict_{}'.format(variable, target)
            x_train['predict_temp'] = (y_train == target).astype(int)  # classification
        prior = x_train['predict_temp'].mean()

        col_avg_y = x_train.groupby(by=variable, axis=0)['predict_temp'].agg([np.mean, np.size])
        col_avg_y["size"] = prior_weight_func(col_avg_y["size"])
        col_avg_y[nf_name] = col_avg_y["size"] * prior + (1 - col_avg_y["size"]) * col_avg_y["mean"]
        col_avg_y.drop(["size", "mean"], axis=1, inplace=True)

        nf_train = x_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = x_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, x, y):
        """
        :param x: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return x_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        x_new = x.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_predict_{}'.format(variable, target): [] for variable, target in
                                  product(self.features, self.target_values)}
            for variable, target in product(self.features, self.target_values):
                nf_name = '{}_predict_{}'.format(variable, target)
                x_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(x, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        x_new.iloc[large_ind], y.iloc[large_ind], x_new.iloc[small_ind],
                        variable, target, self.prior_weight_func)
                    x_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_predict'.format(variable): [] for variable in self.features}
            for variable in self.features:
                nf_name = '{}_predict'.format(variable)
                x_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(x, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        x_new.iloc[large_ind], y.iloc[large_ind], x_new.iloc[small_ind], variable,
                        None, self.prior_weight_func)
                    x_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return x_new

    def transform(self, x):
        """
        :param x: pandas DataFrame, n_samples * n_features
        :return x_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        x_new = x.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.features, self.target_values):
                nf_name = '{}_predict_{}'.format(variable, target)
                x_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    x_new[nf_name] += x_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                x_new[nf_name] /= self.n_splits
        else:
            for variable in self.features:
                nf_name = '{}_predict'.format(variable)
                x_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    x_new[nf_name] += x_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                x_new[nf_name] /= self.n_splits

        return x_new
