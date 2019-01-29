import os
import errno
import pickle  # data loading
import gzip  # data loading
from scipy.stats.stats import pearsonr  # correlation

import numpy as np  # linear algebra
import matplotlib.pyplot as plt  # plotting

# output format
np.set_printoptions(precision=2)
np.seterr(all='raise') # elevate all warnings to errors

def load_mnist_data(path='mnist.pkl.gz'):
    """Loads the data, returns training_data, validation_data, test_data."""
    with gzip.open(path, 'rb') as f:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = pickle.load(f, encoding='latin1')

        return (
          (x_train.reshape((-1,28,28)), y_train),
          (x_val.reshape((-1,28,28)), y_val),
          (x_test.reshape((-1,28,28)), y_test)
        )


# creates random samples for neurons
def create_samples(neuron_count, sample_size):
     return np.random.randint(28, size=(neuron_count, sample_size, 2))
    
# creates "feature vectors" from digits and samples
def digit_to_vectors(digit, pixel_samples):
    return digit[:, pixel_samples[:,:,0],pixel_samples[:,:,1]]
       
# when given images x, labels y, this will collect n digits with label digit
def get_sample_digits(x, y, n, digit):
    positions = np.argwhere(y == digit)[:,0][:n]
    return x[positions]

# helper function - calculates entropies for neurons given average rates on digit class
def calc_entropies(rates):
    # assert first dimension to be 10 digits
    assert(rates.shape[0] == 10)
    
    # normalize 
    normalized = rates / np.sum(rates, axis=0, keepdims=True)
    
    # fix zeros
    normalized[normalized <= 0] = 0.00000001

    # and calc entropies
    return -np.sum(normalized*np.log2(normalized), axis=0)

# if we have recorded rates per neuron and we know the labels of the digits causing this rate
# this function will calculate the mean rate for each digit class using the recorded rates
def mean_rates_per_digit(digit_labels, rates):
    assert(len(digit_labels) == rates.shape[0])
    
    # prepare result array
    result = np.zeros((10, rates.shape[1]))
    
    # visit digits 0 to 9
    for digit in range(10):
        # grab positions of digits
        positions = np.argwhere(digit_labels == digit)[:,0]
        
        # mean rates for these postions
        if len(positions) > 0:
            result[digit] = np.mean(rates[positions], axis=0)
        
    return result

# calculates the input rates and the entropies for given digits and samples
def calc_rates(digits, pixel_samples, standardize_per_digit = True, dampening=10):
    # precalculate activations for these digits
    vectors = digit_to_vectors(digits, pixel_samples)

    # now we sum over the pixel intensities and flatten the array, giving us 1000000 activation values
    activations = np.sum(vectors, axis=2)
    
    if standardize_per_digit:
        # standardize per digit
        means = np.mean(activations, axis=1, keepdims=True)
        stds  = np.std(activations, axis=1, keepdims = True)
        rate_factors = (activations - means) / stds
    else:
        # standardize on all stimuli
        means = np.mean(activations, keepdims=True)
        stds  = np.std(activations, keepdims = True)
        rate_factors = (activations - means) / stds

    # final input rates
    input_rates = rate_factors/dampening + 1
    
    return input_rates

def ensure_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise