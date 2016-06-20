import numpy as np
import numpy.matlib
import numpy.random
from bootstrap import bootstrap
from solver import solve
import matplotlib.pyplot as plt


def simple():
    """
    Simple example with synthetic data, 1 real value + 1 categorical value
    :return:
    """
    n_partitions = 5  # number of dependency classes
    n = n_partitions*500  # number of samples
    x_partition = np.random.uniform(size=n_partitions)
    y_partition = np.array([np.random.binomial(1, x_partition[i])*2-1 for i in range(0, n_partitions)])  # binary label
    partitions = np.matlib.repmat(range(0, n_partitions), 1, np.floor(n/n_partitions))[0]
    corruption = np.random.choice(range(0, n_partitions), size=n)
    corruption_selector = np.random.binomial(1, 0.2, size=n)
    partitions_corrupted = partitions*(1-corruption_selector) + corruption*corruption_selector

    X = np.matlib.repmat(x_partition, 1, np.floor(n / n_partitions))[0] + np.random.normal(scale=0.3, size=n)
    y = np.matlib.repmat(y_partition, 1, np.floor(n / n_partitions))[0]
    corruption = np.random.choice(range(0, n_partitions), size=n)
    corruption_selector = np.random.binomial(1, 0.2, size=n)
    categorical = partitions * (1 - corruption_selector) + corruption * corruption_selector
    categorical_full = np.zeros((n, n_partitions))
    categorical_full[np.array(range(0, categorical.size)), categorical] = 1
    X = np.hstack((X[:, None], categorical_full))
    test_class = np.argsort(x_partition)[np.floor(n_partitions/2)]
    X_train = X[partitions_corrupted != test_class, :]  # some samples from test class will end up in train class
    y_train = y[partitions_corrupted != test_class]
    X_test = X[np.all([partitions_corrupted == test_class, partitions == test_class], axis=0), :]  # no train samples in test class
    y_test = y[np.all([partitions_corrupted == test_class, partitions == test_class], axis=0)]

    additional_corruption_levels = [0, y_test.size/8, y_test.size/4, y_test.size/8*3, y_test.size/2]
    B = bootstrap(X_train, X_test, y_train, y_test, additional_corruption_levels)
    orig = bootstrap(X[partitions != test_class, :], X[partitions == test_class, :], y[partitions != test_class], y[partitions == test_class], [0])
    [_, n_bins] = B.shape
    s = np.sum(np.all([partitions_corrupted != test_class, partitions == test_class]))
    #C, s = solve(B, additional_corruption_levels, y_train.size, s)
    #[max_solved_corruption_levels, _] = C.shape

    #plt.figure(1)
     #plt.title('LOCCO')

    #plt.subplot(311)
    plt.plot(np.linspace(0, 1, num=n_bins), np.vstack((orig, B[np.array([0, B.shape[0]-1]), :])).T, linewidth=3)
    plt.xlabel('Error')
    plt.ylabel('Density')
    axes = plt.gca()
    #axes.set_ylim([0, 20])
    plt.legend(['Zero Corruption', 'Observed Corruption', 'Maximum Artificial Corruption'])
    plt.show()
    # plt.title('Error PDFs with increasing test-->train corruption')

    # plt.subplot(312)
    # plt.plot(np.linspace(0, 1, num=n_bins), C[0:3, :].T)  # np.vstack((orig[0, :], B[0, :], C[0:3, :])).T)
    # plt.xlabel('Error')
    # axes = plt.gca()
    # axes.set_ylim([0, 20])
    # plt.ylabel('Density')
    # plt.legend(['Estimate of Uncorrupted', 'Estimate s=1', 'Estimate s=2'])
    #
    # plt.subplot(313)
    # plt.plot(np.linspace(0, 1, num=n_bins), np.cumsum(C[0:3, :], axis=1).T)  # np.vstack((orig[0, :], B[0, :], C[0:3, :])).T)
    # plt.xlabel('Error')
    # axes = plt.gca()
    # axes.set_ylim([0, 20])
    # plt.ylabel('Cumulative Density')
    # plt.legend(['Estimate of Uncorrupted', 'Estimate s=1', 'Estimate s=2'])
    # plt.show()
    # print('Finished')

if __name__ == '__main__':
    simple()