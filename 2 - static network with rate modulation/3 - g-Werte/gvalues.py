import nest
import sys
import os.path
import numpy as np
from math import degrees
import matplotlib.pyplot as plt

sys.path.append('../')
import params
import nest_tools
import mnist_tools

# reseed numpy
np.random.seed(0)

# create samples

# load data - some preprocessing is done in the module (reshaping)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_tools.load_mnist_data('../mnist.pkl.gz')

# create pixel samples
pixel_samples = mnist_tools.create_samples(4000, sample_size=40)

# create training set
digits = np.concatenate([ mnist_tools.get_sample_digits(x_train, y_train, 10, i) for i in range(100) ])
digit_labels = np.concatenate([ [i//10] for i in range(100) ])

def simulate(rates, digits, digit_labels, name, duration=1000.0):
    network = nest_tools.Network()
    network.reset_nest()
    network.setup_static_network()
    network.record_spikes(name)
    
    for i, digit in enumerate(digits):
        # set rate for every neuron

        for j, rate in enumerate(rates[i]):
            network.set_rate([j+1], rate)

        print(str(i+1)+". stimulus = "+str(digit_labels[i])+", simulating for", duration)
        nest.Simulate(duration)

    network.save_recording(name)

input_rates = mnist_tools.calc_rates(digits, pixel_samples, standardize_per_digit=True) * params.rate
mean_input_rates_per_digit =  mnist_tools.mean_rates_per_digit(digit_labels, input_rates)
input_entropies = mnist_tools.calc_entropies(mean_input_rates_per_digit)

for g in [2, 5, 8, 11]:
    params.g = float(g)
    if not os.path.exists('result'+str(g)+'.0.npy'):
        simulate(input_rates, digits, digit_labels, 'result'+str(g))
    
#    recording = nest_tools.SpikeRecording.from_file('result'+str(g))

#    output_rates = np.stack([recording.rate(range(1,4001), i*1000, (i+1)*1000.0) for i in range(len(digits))])
#    mean_output_rates_per_digit = mnist_tools.mean_rates_per_digit(digit_labels, output_rates)
#    output_entropies = mnist_tools.calc_entropies(mean_output_rates_per_digit)