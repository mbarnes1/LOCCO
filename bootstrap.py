import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing
from functools import partial


def bootstrap(X_train, y_train, X_test, y_test, X_for_corruption, y_for_corruption, prob_additional_corruption, nx_boot, n_processes=1):
    """
    Bootstrapped train/testing
    :param X_train: Array of [nsamples x nfeatures]. May include categorical fields
    :param y_train: Vector of labels
    :param X_test: Array of [nsamples x nfeatures]. May include categorical fields
    :param y_test: Vector of labels
    :param X_for_corruption: Array of [nsamples x nfeatures]. May include categorical fields
    :param y_for_corruption: Vector of labels
    :param prob_additional_corruption: Probability of adding additional corruption
    :param nx_boot: Number of samples to take in bootstrap
    :return b: Observed mean error vector, ncorruption x 1
    :return B: Observed error PDF's, array ncorruptions x nbins
    """
    pool = multiprocessing.Pool(n_processes)
    n_bins = 25

    A_hat = np.empty((len(prob_additional_corruption), nx_boot + 1))
    b = np.empty(len(prob_additional_corruption))
    B = np.empty((len(prob_additional_corruption), n_bins))
    x_hat = np.zeros(nx_boot + 1)

    X_train_corrupted = np.vstack((X_train, X_for_corruption))
    y_train_corrupted = np.hstack((y_train, y_for_corruption))
    [n_train, _] = X_train.shape
    [n_for_corruption, _] = X_for_corruption.shape

    function = partial(bootstrap_fixed_corruption, X_train_corrupted=X_train_corrupted, y_train_corrupted=y_train_corrupted, X_test=X_test, y_test=y_test, nx_boot=nx_boot, n_for_corruption=n_for_corruption, n_train=n_train, n_bins=n_bins)
    result = pool.map(function, prob_additional_corruption)
    for i, r in enumerate(result):
        b[i] = r[0]
        B[i, ] = r[1]
        A_hat[i, ] = r[2]
        x_hat += r[3]
    x_hat = x_hat/np.sum(A_hat, axis=0)
    x_hat = np.nan_to_num(x_hat)
    A_hat = (A_hat.T/np.sum(A_hat, axis=1)).T
    return b, B, A_hat, x_hat


def bootstrap_fixed_corruption(corruption, X_train_corrupted, y_train_corrupted, X_test, y_test, nx_boot, n_for_corruption, n_train, n_bins):
    """

    :return:
    """
    boots = 2000
    a_hat = np.zeros(nx_boot+1)
    x_hat = np.zeros(nx_boot+1)
    classifier = KNeighborsClassifier()
    errors = []
    p_vector = np.hstack(
        (np.full(n_train, (1.0 - corruption) / n_train), np.full(n_for_corruption, corruption / n_for_corruption)))
    for j in range(0, boots):
        ind_train_boot = np.random.choice(n_train + n_for_corruption, size=nx_boot, replace=True, p=p_vector)
        s = np.sum(ind_train_boot >= n_train)
        a_hat[s] += 1
        X_train_boot = X_train_corrupted[ind_train_boot, :]
        y_train_boot = y_train_corrupted[ind_train_boot]
        classifier.fit(X_train_boot, y_train_boot)
        error = 1 - classifier.score(X_test, y_test)
        errors.append(error)
        x_hat[s] += error
    b = np.mean(errors)
    [pdf, _] = np.histogram(errors, bins=n_bins, range=(0, 1), density=True)
    B = pdf
    return b, B, a_hat, x_hat