from scipy.stats import binom
import numpy as np
import cvxpy


def solve_means(b, probabilities_artifical_corruption, probability_natural_corruption, nx_boot):
    """
    Estimate the true error means in LOCCO
    :param b: Observed error means [delta_s.size x 1]
    :param probabilities_artifical_corruption: Vector of probability of additional corruption added to training set
    :param probability_natural_corruption: Probability of sampling a corrupted sample in the training set
    :param nx_boot: Number of samples taken (with replacement) from train set (with natural and artifical corruption)
    :return x: Vector of estimated mean errors, [delta_s.size + s  x  1]
    """
    A = np.array([binom.pmf(range(0, nx_boot), nx_boot, (1-p)*probability_natural_corruption + p) for p in probabilities_artifical_corruption])
    x = cvxpy.Variable(nx_boot)
    obj = cvxpy.Minimize(cvxpy.norm(A * x - b))

    off_diagonal = np.diag(np.ones(nx_boot - 1), 1)[0:-1, :]
    diagonal = np.eye(nx_boot)[0:-1, :]
    #constraints = [x >= 0,
    #               x <= 1]
    constraints = [x[-1] >= 0,
                   x[0] <= 1,
                   off_diagonal*x <= diagonal*x]
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(verbose=True, reltol=1e-10)
    print "status:", prob.status
    print "optimal value", prob.value

    return np.array(x.value), A


def solve_pdfs(B, delta_s, n, s):
    """
    Estimates the true error distribution in LOCCO
    :param B: Observed error histograms [delta_s.size x nbins]
    :param delta_s: Additional corruption added to training set
    :param n: Number of training samples (before adding additional corruption)
    :param s: If original corruption is known
    :return X: Error histograms for increasing levels of corruption [n_corruption x error]
    """
    #TODO Fix this so format is like solve_means
    n_corruption = s + max(delta_s) + 1  # include zero corruption
    [_, n_bins] = B.shape
    A = np.array([binom.pmf(range(0, n_corruption), s+d_s, float(s+d_s)/(n+d_s)) for d_s in delta_s])
    X = cvxpy.Variable(n_corruption, n_bins)
    obj = cvxpy.Minimize(cvxpy.norm(A*X-B, "fro"))
    constraints = [X*np.ones((n_bins, 1)) == n_bins*np.ones((n_corruption, 1))]
    off_diagonal = np.diag(np.ones(n_corruption-1), 1)[0:-1, :]
    diagonal = np.eye(n_corruption)[0:-1, :]
    indicator = np.zeros((n_bins, 1))
    for i in range(0, n_bins):
        indicator[i, 0] = 1
        constraints.append(off_diagonal*X*indicator <= diagonal*X*indicator)
    prob = cvxpy.Problem(obj, constraints)
    prob.solve()
    print "status:", prob.status
    print "optimal value", prob.value
    print "optimal var", X.value

    return np.array(X.value)
