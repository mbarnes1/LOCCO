import numpy as np
import numpy.matlib
import numpy.random
from bootstrap import bootstrap
from solver import solve_pdfs, solve_means
import matplotlib.pyplot as plt
import cPickle as pickle
from scipy.stats import binom


def simple():
    """
    Simple example with synthetic data, 1 real value + 1 categorical value
    :return:
    """
    # User Defined Parameters #
    n_partitions = 5
    n_per_partition = 200
    partition_corruption_level = 0.05
    categorical_corruption_level = 0.2
    percent_test_for_corruption = 0.5
    min_prob_sampling_zero_corruption = 0.2
    n_processes = 4
    #percent_train_bootstrap = 0.2

    # Synthetic Data #
    X_train_uncorrupted, y_train_uncorrupted, X_train_corrupted, y_train_corrupted, X_test, y_test, X_for_corruption, y_for_corruption, s = draw_synthetic(n_partitions, n_per_partition, partition_corruption_level, categorical_corruption_level, percent_test_for_corruption)
    #n_train = y_train_corrupted.size

    # Bootstrapping #
    nx_boot = 100  #int(np.log(min_prob_sampling_zero_corruption)/np.log(float(n_train - s)/n_train))
    print 'Bootstrapping with ', nx_boot, ' samples from train set'
    prob_add_corruption = np.linspace(start=0.05, stop=0.3, num=nx_boot)

    [b, B, A_hat, x_hat] = bootstrap(X_train_uncorrupted, y_train_uncorrupted, X_test, y_test, X_for_corruption, y_for_corruption, prob_add_corruption, nx_boot, n_processes=n_processes)
    orig, _, _, _ = bootstrap(X_train_uncorrupted, y_train_uncorrupted, X_test, y_test, X_for_corruption, y_for_corruption, [0], nx_boot)  # X[partitions_uncorrupted == test_class, :], y[partitions_uncorrupted == test_class], X, y, [0], nx_boot)
    s = 0
    n_train = len(y_train_uncorrupted)

    # Solve for uncorrupted error #
    pickle.dump([b, prob_add_corruption, n_train, s, orig, nx_boot, A_hat, x_hat], open('bootstrap.p', 'wb'))
    #[b, prob_add_corruption, n_train, s, orig, nx_boot, A_hat, x_hat] = pickle.load(open('bootstrap.p', 'rb'))
    x_hat = np.nan_to_num(x_hat)
    print 'Observed mean errors', b
    print 'True mean error', orig
    mean_errors, A = solve_means(b, prob_add_corruption, n_train, s, nx_boot)
    print "Solved mean errors"
    print mean_errors


    percent_corruption = [(1-p)*float(s)/n_train + p for p in prob_add_corruption]
    percent_corruption_all = np.linspace(start=0, stop=1, num=250)
    plt.plot(A.T)
    plt.plot(A_hat.T, '--')
    plt.xlabel('Corruption')
    plt.ylabel('Frequency')
    plt.show()

    plt.plot(mean_errors)
    plt.plot(x_hat)
    plt.xlabel('Corruption')
    plt.ylabel('Mean Error')
    plt.legend(['Optimal, solved', 'Observed'])
    plt.show()

    A_all = np.array([binom.pmf(range(0, nx_boot), nx_boot, p) for p in percent_corruption_all])
    plt.plot(percent_corruption, b, 'g')
    plt.plot(percent_corruption_all, A_all*np.matrix(mean_errors), 'b--')
    plt.plot(percent_corruption, np.dot(A_hat, x_hat), 'r--')
    plt.plot(0, orig, 'or')
    plt.plot(0, mean_errors[0], 'ob')
    plt.xlim([-0.1, max(percent_corruption)+0.1])
    plt.ylim([0, max(orig, mean_errors[0])*1.1])
    plt.legend(['Observed b', 'Solved A*x_star', 'Observed A_hat*x_hat', 'True Error', 'Estimated Error'])
    plt.xlabel('Corruption')
    plt.ylabel('Error')
    plt.show()


def draw_synthetic(n_partitions, n_per_partition, partition_corruption_level, categorical_corruption_level, percent_test_for_corruption):
    """

    :return:
    """
    n = n_partitions * n_per_partition  # number of samples

    partitions_uncorrupted = np.matlib.repmat(range(0, n_partitions), 1, np.floor(n / n_partitions))[0]
    partition_corruption = np.random.choice(range(0, n_partitions), size=n)
    corruption_selector = np.random.binomial(1, partition_corruption_level, size=n)
    partitions_corrupted = partitions_uncorrupted * (
    1 - corruption_selector) + partition_corruption * corruption_selector

    x1_partition = np.random.uniform(size=n_partitions)
    x1 = np.matlib.repmat(x1_partition, 1, np.floor(n / n_partitions))[0] + np.random.normal(scale=0.3, size=n)

    corruption_selector = np.random.binomial(1, categorical_corruption_level, size=n)
    x2_int = partitions_uncorrupted * (1 - corruption_selector) + partition_corruption * corruption_selector
    x2_full = np.zeros((n, n_partitions))
    x2_full[np.array(range(0, x2_int.size)), x2_int] = 1
    X = np.hstack((x1[:, None], x2_full))

    y_partition = np.array(
        [np.random.binomial(1, x1_partition[i]) * 2 - 1 for i in range(0, n_partitions)])  # binary label
    y = np.matlib.repmat(y_partition, 1, np.floor(n / n_partitions))[0]

    # Train/test/corruption split #
    test_class = np.argsort(x1_partition)[np.floor(n_partitions / 2)]
    X_train_uncorrupted = X[partitions_uncorrupted != test_class, :]
    y_train_uncorrupted = y[partitions_uncorrupted != test_class]
    X_train_corrupted = X[partitions_corrupted != test_class, :]  # some samples from test class will end up in train class
    y_train_corrupted = y[partitions_corrupted != test_class]
    X_test_original = X[np.all([partitions_corrupted == test_class, partitions_uncorrupted == test_class], axis=0),
                      :]  # no train samples in test class
    y_test_original = y[np.all([partitions_corrupted == test_class, partitions_uncorrupted == test_class], axis=0)]
    ind_for_corruption = np.random.choice(range(0, y_test_original.size),
                                          size=np.floor(y_test_original.size * percent_test_for_corruption))
    X_for_corruption = X_test_original[ind_for_corruption, :]
    y_for_corruption = y_test_original[ind_for_corruption]
    X_test = np.delete(X_test_original, ind_for_corruption, axis=0)
    y_test = np.delete(y_test_original, ind_for_corruption, axis=0)

    s = np.sum(np.all(np.array([partitions_corrupted != test_class, partitions_uncorrupted == test_class]), axis=0))
    return X_train_uncorrupted, y_train_uncorrupted, X_train_corrupted, y_train_corrupted, X_test, y_test, X_for_corruption, y_for_corruption, s

if __name__ == '__main__':
    simple()