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
pixel_samples = mnist_tools.create_samples(4000, sample_size=40)

network = nest_tools.Network(plasticity=True, target_rate=8.0/1000)
network.reset_nest(print_time=False)
network.setup_static_network()
network.record_spikes_to_file()

# create training set
last = np.zeros((4000,4000))
for t in range(5):
    print("Timestep", t)
    nest.Simulate(1000)

    # network.save_recording("recording", "data/spike_recording"+str(t))
    # network.reset_recording("recording")

    # save matrices
    current = network.snapshot_connectivity_matrix()
    
    # save change matrix, but sparsified
    change = current - last
    sparsified = sparse.csr_matrix(change)
    
    np.save("data/matrix_change"+str(t)+"."+str(nest.Rank()), sparsified)

    # update last variable
    last = current
    
np.save("data/final_connectivity."+str(nest.Rank()), current)
