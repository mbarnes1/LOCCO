import numpy as np
import numpy.random
from bootstrap import bootstrap
from solver import solve_pdfs, solve_means
import cPickle as pickle
from scipy.stats import binom
from distributions import Simple


def simple():
    """
    Simple example with synthetic data, 1 real value + 1 categorical value
    :return:
    """

    # User Defined Parameters #
    number_partitions = 10
    categorical_corruption_probability = 0.2
    min_prob_sampling_zero_corruption = 0.2
    number_processes = 50
    nx_boot = 50  # int(np.log(min_prob_sampling_zero_corruption)/np.log(float(n_train - s)/n_train))
    probabilities_artificial_corruption = np.linspace(start=0.02, stop=0.3, num=nx_boot)

    # Synthetic Data #
    distribution = Simple(number_partitions=number_partitions,
                          categorical_corruption_probability=categorical_corruption_probability)
    partitions = distribution.get_partitions()
    train_partitions = np.random.choice(partitions, size=len(partitions)*0.8, replace=False)
    test_partitions = np.array(list(set(partitions).difference(set(train_partitions))))

    # Bootstrapping #
    print 'Bootstrapping with ', nx_boot, ' samples'
    s = 0
    [b, B, A_hat, x_hat] = bootstrap(distribution, train_partitions=train_partitions, test_partitions=test_partitions,
                                     probabilities_artificial_corruption=probabilities_artificial_corruption, nx_boot=nx_boot,
                                     number_processes=number_processes)
    orig, _, _, _ = bootstrap(distribution, train_partitions=train_partitions, test_partitions=test_partitions,
                              probabilities_artificial_corruption=[0.0], nx_boot=nx_boot, number_processes=number_processes)

    # Solve for uncorrupted error #
    probability_natural_corruption = 0.0
    #pickle.dump([b, probabilities_artificial_corruption, probability_natural_corruption, orig, nx_boot, A_hat, x_hat], open('bootstrap.p', 'wb'))
    [b, probabilities_artificial_corruption, probability_natural_corruption, orig, nx_boot, A_hat, x_hat] = pickle.load(open('bootstrap.p', 'rb'))
    x_hat = np.nan_to_num(x_hat)
    print 'Observed mean errors', b
    print 'True mean error', orig
    mean_errors, A = solve_means(b=b, probabilities_artifical_corruption=probabilities_artificial_corruption,
                                 probability_natural_corruption=probability_natural_corruption, nx_boot=nx_boot)
    print "Solved mean errors"
    print mean_errors

    # Make plots
    import matplotlib.pyplot as plt
    percent_corruption = [(1-p)*probability_natural_corruption + p for p in probabilities_artificial_corruption]
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

if __name__ == '__main__':
    simple()