import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def bootstrap(X_train, X_test, y_train, y_test, delta_s):
    """
    Bootstrapped train/testing
    :param X_train: Array of [nsamples x nfeatures]. May include categorical fields
    :param X_test: Array of [nsamples x nfeatures]. May include categorical fields
    :param y_train: Vector of labels
    :param y_test: Vector of labels
    :param delta_s: Corruption levels to add
    :return B: Observed error PDF's, array ncorruptions x nbins
    """
    n_boot = 1000
    n_bins =  25
    classifier = KNeighborsClassifier()

    # Can I do a double bootstrap? Or should I hold y_test fixed?
    [n_train, _] = X_train.shape
    [n_test, _] = X_test.shape
    p = np.floor(n_train/10)
    ind_test = np.full(n_test, False, dtype=bool)
    ind_test[np.random.choice(range(0, n_test), n_test - np.max(delta_s), replace=True)] = True
    X_for_corruption = X_test[~ind_test, :]
    y_for_corruption = y_test[~ind_test]
    X_test = X_test[ind_test, :]
    y_test = y_test[ind_test]
    B = []
    for corruption in delta_s:
        X_train_corrupted = np.vstack((X_train, X_for_corruption[0:corruption, :]))
        y_train_corrupted = np.hstack((y_train, y_for_corruption[0:corruption]))
        errors = []
        for i in range(0, n_boot):
            ind_boot = np.random.randint(low=0, high=(n_train + corruption), size=p)
            X_train_boot = X_train_corrupted[ind_boot, :]
            y_train_boot = y_train_corrupted[ind_boot]
            classifier.fit(X_train_boot, y_train_boot)
            errors.append(1-classifier.score(X_test, y_test))
        [pdf, _] = np.histogram(errors, bins=n_bins, range=(0, 1), density=True)
        B.append(pdf)
    B = np.array(B)
    return B

