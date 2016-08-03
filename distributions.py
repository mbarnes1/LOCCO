import numpy as np
import pandas as pd
import string
from itertools import izip


class Simple(object):
    def __init__(self, number_partitions, categorical_corruption_probability=0.2):
        """
        Simple distribution with 1 numerical feature and 1 categorical feature
        :param number_partitions:
        :param categorical_corruption_probability:
        """
        self._number_partitions = number_partitions
        self._categorical_corruption_probability = categorical_corruption_probability
        x_numerical = np.sort(np.random.uniform(size=self._number_partitions))
        categories = list(string.ascii_lowercase[0:self._number_partitions])
        x_categorical = pd.Series(categories)
        x_categorical = x_categorical.astype('category', categories=categories)
        y = [np.random.binomial(1, x_numerical[i]) * 2 - 1 for i in range(0, number_partitions)]
        data = {'numerical': x_numerical, 'categorical': x_categorical, 'label': y}
        self._partition_centers = pd.DataFrame.from_dict(data=data, orient='columns')

    def sample(self, number_samples, partitions=None):
        """
        Sample from the distribution
        :param number_samples: Integer, number of samples to draw
        :param partitions: Partitions to sample from (default is all), a list
        :return samples: Panda object
        """
        if partitions is None:
            partitions = self.get_partitions()
        sampled_partitions = np.random.choice(partitions, size=number_samples, replace=True)
        samples = self._partition_centers.loc[sampled_partitions]
        samples.reset_index(inplace=True)
        samples = samples.rename({'index': 'true partition'})

        numerical_noise = np.random.normal(scale=0.3, size=number_samples)
        samples = samples.set_value(samples.index, 'numerical', samples['numerical'] + numerical_noise)

        number_samples_to_corrupt_categories = np.random.binomial(number_samples,
                                                                  p=self._categorical_corruption_probability)
        indices_to_corrupt = np.random.choice(samples.index, size=number_samples_to_corrupt_categories)
        corrupted_categories = np.random.choice(self._partition_centers['categorical'],
                                                size=number_samples_to_corrupt_categories, replace=True)
        for i, c in izip(indices_to_corrupt, corrupted_categories):
            samples.set_value(i, 'categorical', c)

        return samples

    def get_partitions(self):
        """
        Returns indices of all partitions
        :return partitions: List of all partitions
        """
        return self._partition_centers.index


def sample_mix(samplers, probabilities, n):
    """
    Sample from mixture of distributions
    :param samplers: List of sampling functions, with single argument for number of samples to return
    :param probabilities: List of probabilities of sampling from each distribution (must sum to 1, same length as distributions)
    :param n: Number of samples to return
    :return samples: Samples, as a pandas object
    :return n_d: List of number of samples from each distribution (sums to n)
    """
    n_d = np.random.multinomial(n, probabilities, size=1)[0]
    samples = []
    for sampler, n_s in izip(samplers, n_d):
        if n_s > 0:
            samples.append(sampler(n_s))
    samples = pd.concat(samples)
    samples.reset_index(inplace=True)
    del samples['index']
    return samples, n_d


