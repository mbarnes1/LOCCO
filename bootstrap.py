import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing
from functools import partial
from distributions import sample_mix
import pandas as pd


def bootstrap(distribution, train_partitions, test_partitions, probabilities_artificial_corruption, nx_boot,
              number_processes=1):
    """
    Bootstrapped train/testing
    :param distribution: Object with sample(n, partitions) method
    :param train_partitions: List of partitions in the true train set
    :param test_partitions: List of partitions in the true test set
    :param probabilities_artificial_corruption: Probability of adding additional corruption
    :param nx_boot: Number of samples to take in bootstrap
    :param number_processes: Number of processes to use
    :return b: Observed mean error vector, ncorruption x 1
    :return B: Observed error PDF's, array ncorruptions x nbins
    """
    pool = multiprocessing.Pool(number_processes)
    n_bins = 25

    A_hat = np.empty((len(probabilities_artificial_corruption), nx_boot + 1))
    b = np.empty(len(probabilities_artificial_corruption))
    B = np.empty((len(probabilities_artificial_corruption), n_bins))
    x_hat = np.zeros(nx_boot + 1)

    function = partial(bootstrap_fixed_corruption, distribution=distribution, train_partitions=train_partitions,
                       test_partitions=test_partitions, samples_in_bootstrap=nx_boot, n_bins=n_bins)
    result = [function(x) for x in probabilities_artificial_corruption]  # pool.map(function, probabilities_artificial_corruption)
    for i, r in enumerate(result):
        b[i] = r[0]
        B[i, ] = r[1]
        A_hat[i, ] = r[2]
        x_hat += r[3]
    x_hat = x_hat/np.sum(A_hat, axis=0)
    x_hat = np.nan_to_num(x_hat)
    A_hat = (A_hat.T/np.sum(A_hat, axis=1)).T
    return b, B, A_hat, x_hat


def bootstrap_fixed_corruption(probability_artificial_corruption, distribution, train_partitions, test_partitions, samples_in_bootstrap, n_bins):
    """
    TODO
    :return:
    """
    sampler_train = partial(distribution.sample, partitions=train_partitions)
    sampler_test = partial(distribution.sample, partitions=test_partitions)
    sampler_with_artificial_corruption = partial(sample_mix, samplers=[sampler_train, sampler_test],
                                                 probabilities=[1.0-probability_artificial_corruption,
                                                                probability_artificial_corruption])
    boots = 100
    a_hat = np.zeros(samples_in_bootstrap + 1)
    x_hat = np.zeros(samples_in_bootstrap + 1)
    classifier = KNeighborsClassifier()
    errors = []

    for j in range(0, boots):
        # Sample data
        samples_train, n_d = sampler_with_artificial_corruption(n=samples_in_bootstrap)
        s = n_d[1]
        a_hat[s] += 1
        X_train = pd.concat([samples_train['numerical'], pd.get_dummies(samples_train['categorical'], prefix='cat_')], axis=1)
        y_train = samples_train['label']
        classifier.fit(X_train, y_train)

        samples_test = sampler_test(1000)
        X_test = pd.concat([samples_test['numerical'], pd.get_dummies(samples_test['categorical'], prefix='cat_')], axis=1)
        y_test = samples_test['label']
        error = 1 - classifier.score(X_test, y_test)
        errors.append(error)
        x_hat[s] += error
    b = np.mean(errors)
    [pdf, _] = np.histogram(errors, bins=n_bins, range=(0, 1), density=True)
    B = pdf
    return b, B, a_hat, x_hat