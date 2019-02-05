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
from scipy import sparse
import pickle  # data loading
import gzip  # data loading

# reseed numpy
np.random.seed(0)

# load data - some preprocessing is done in the module (reshaping)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_tools.load_mnist_data('../mnist.pkl.gz')

# create pixel samples
pixel_samples = mnist_tools.create_samples(2000, sample_size=40)

network = nest_tools.Network(plasticity=True, target_rate=8.0/1000)
network.reset_nest(print_time=False)
network.setup_static_network()

# initialize spike detector
network.record_spikes("recording")

# connectivity matrix
last = np.zeros((5000,5000))

INIT_DURATION = 300
DIGITS = 5000
POST_STIM = 1000

# open a gzipped file to record all data...
with gzip.open('data/snapshots'+str(nest.Rank()), 'wb') as f:
    for t in range(INIT_DURATION):
        print("Timestep", t)
        nest.Simulate(1000)

        # spikes from recording, dump them...
        times, senders = network.spikes("recording")
        pickle.dump(times, f)
        pickle.dump(senders, f)

        # flush all spike data
        network.reset_recording("recording")

        # save matrices
        current = network.snapshot_connectivity_matrix()
        
        # save change matrix, but sparsified
        change = current - last
        sparsified = sparse.csr_matrix(change)
        pickle.dump(sparsified, f)

        # update last variable
        last = current
    
    for t in range(DIGITS):
        input_rates = mnist_tools.calc_rates(x_train[t:t+1], pixel_samples, standardize_per_digit=True) * params.rate
        input_rates = input_rates.squeeze()

        #print(input_rates.shape, np.mean(input_rates))

        for j, rate in enumerate(input_rates):
            network.set_rate([j+1], rate)

        teacher_stim_index = 2000+y_train[t]*200

        for j in range(2000,  4000):
            network.set_rate([j+1], params.rate)

        for j in range(teacher_stim_index,  teacher_stim_index+200):
            network.set_rate([j+1], 1.1 * params.rate)

        print("Nr.", t, "Digit", y_train[t])
        nest.Simulate(1000)

        # spikes from recording, dump them...
        times, senders = network.spikes("recording")
        pickle.dump(times, f)
        pickle.dump(senders, f)

        # flush all spike data
        network.reset_recording("recording")

        # save matrices
        current = network.snapshot_connectivity_matrix()
        
        # save change matrix, but sparsified
        change = current - last
        sparsified = sparse.csr_matrix(change)
        pickle.dump(sparsified, f)

        # update last variable
        last = current

    # after learning, turn of plasticity and teacher stimulus

    nest.DisableStructuralPlasticity()
    for j in range(2000, 4000):
        network.set_rate([j+1], params.rate)

    for t in range(DIGITS, DIGITS+POST_STIM):
        input_rates = mnist_tools.calc_rates(x_train[t:t+1], pixel_samples, standardize_per_digit=True) * params.rate
        input_rates = input_rates.squeeze()

        for j, rate in enumerate(input_rates):
            network.set_rate([j+1], rate)

        print("poststim Nr.", t, "Digit", y_train[t])
        nest.Simulate(1000)

        # spikes from recording, dump them...
        times, senders = network.spikes("recording")
        pickle.dump(times, f)
        pickle.dump(senders, f)

        # flush all spike data
        network.reset_recording("recording")

        # save matrices
        current = network.snapshot_connectivity_matrix()
        
        # save change matrix, but sparsified
        change = current - last
        sparsified = sparse.csr_matrix(change)
        pickle.dump(sparsified, f)

        # update last variable
        last = current
    

np.save("data/final_connectivity."+str(nest.Rank()), sparse.csr_matrix(current))
