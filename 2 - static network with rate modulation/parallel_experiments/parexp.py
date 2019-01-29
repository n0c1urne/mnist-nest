import nest
import numpy as np
import params
from math import degrees
import matplotlib.pyplot as plt
from nest_tools import Network, SpikeRecording
import mnist_tools

# reseed numpy
np.random.seed(0)

mnist_tools.ensure_path('parallel_experiments')

# load data - some preprocessing is done in the module (reshaping)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_tools.load_mnist_data()


def simulate(rates, digits, digit_labels, filename, duration=1000.0):
    network = Network()
    network.reset_nest()

    network.setup_static_network()
    network.record_spikes(filename)

    for i, digit in enumerate(digits):
        # set rate for every neuron

        for j, rate in enumerate(rates[i]):
            network.set_rate([j+1], rate)

        print(str(i+1)+". stimulus = "+str(digit_labels[i])+", simulating for", duration)
        nest.Simulate(duration)

    network.save_recording(filename)

# create training set
#digits = np.concatenate([ mnist_tools.get_sample_digits(x_train, y_train, 1, i) for i in range(10) ])
#digit_labels = np.concatenate([ [i] for i in range(10) ])
digits = np.concatenate([ mnist_tools.get_sample_digits(x_train, y_train, 10, i) for i in range(10) ])
digit_labels = np.concatenate([ [i//10] for i in range(100) ])



parameters = [ (rate, samples) for samples in [20, 40, 80] for rate in [10, 20]]


for rate, samples in parameters:
    # samples for each neuron
    pixel_samples = mnist_tools.create_samples(4000, sample_size=samples)
        
    input_rates = mnist_tools.calc_rates(digits, pixel_samples, standardize_per_digit=True, dampening=rate) * params.rate
    input_rates[input_rates < 0] = 0.0
    mean_input_rates_per_digit =  mnist_tools.mean_rates_per_digit(digit_labels, input_rates)
    input_entropies = mnist_tools.calc_entropies(mean_input_rates_per_digit)
        
    simulate(input_rates, digits, digit_labels, 'parallel_experiments/experiment')

    # plot every 10th neuron
    if nest.Rank() == 0:
        recording = SpikeRecording.from_file('parallel_experiments/experiment')

        output_rates = np.stack([recording.rate(range(1,4001), i*1000, (i+1)*1000.0) for i in range(len(digits))])
        mean_output_rates_per_digit =  mnist_tools.mean_rates_per_digit(digit_labels, output_rates)
        output_entropies = mnist_tools.calc_entropies(mean_output_rates_per_digit)

        plt.figure(figsize=(15,15))
        plt.title("test")
        
        plt.subplot(3,2,1)
        plt.title("input rate variation")
        plt.hist(input_rates.ravel()/15000, bins=200)

        plt.subplot(3,2,2)
        plt.title("output rates")
        plt.hist(output_rates.ravel(), bins=200)
        
        plt.subplot(3,2,3)
        plt.title("input entropies")
        plt.hist(input_entropies, bins=200)

        plt.subplot(3,2,4)
        plt.title("output entropies")
        plt.hist(output_entropies, bins=200)

        plt.subplot(3,2,5)
        plt.scatter(input_entropies, output_entropies, s=0.2)
        plt.xlabel("input entropy")
        plt.ylabel("output entropy")
        plt.plot([0,4], [0,4])
        plt.savefig('parallel_experiments/res-'+str(rate)+'-'+str(samples))

    
