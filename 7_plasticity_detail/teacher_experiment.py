import nest
import sys
import os.path
import numpy as np
from math import degrees
import matplotlib.pyplot as plt
import argparse

sys.path.append('../')
import params
import nest_tools
import mnist_tools
from scipy import sparse
import pickle  # data loading
import gzip  # data loading


def save_data(f, last, network):
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

    return current



def simulation(name, teacher_strength, stimulus_duration):
    # load data - some preprocessing is done in the module (reshaping)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_tools.load_mnist_data('../mnist.pkl.gz')
    
    if not os.path.exists(name):
        try:
            os.makedirs(name)
        except:
            pass

    # reseed numpy
    np.random.seed(0)
    network = nest_tools.Network(plasticity=True, target_rate=8.0/1000, prewire=True)
    
    network.reset_nest(print_time=False)
    network.setup_static_network()

    # initialize spike detector
    network.record_spikes("recording")

    # connectivity matrix
    last = np.zeros((5000,5000))

    INIT_DURATION = 100
    DIGITS = 1
    POST_STIM = 300

    # open a gzipped file to record all data...
    with gzip.open(name+'/snapshots'+str(nest.Rank()), 'wb') as f:
        for t in range(INIT_DURATION):
            print("Timestep", t)
            nest.Simulate(1000)

            # spikes from recording, dump them...
            last = save_data(f, last, network)

        for t in range(DIGITS):
            teacher_stim_index = 0

            for j in range(2000,  4000):
                network.set_rate([j+1], params.rate)

            for j in range(teacher_stim_index,  teacher_stim_index+400):
                network.set_rate([j+1], (1.0 + teacher_strength/100.0) * params.rate)

            print("Nr.", t, "Digit", y_train[t])

            for i in range(stimulus_duration):
                nest.Simulate(1000)

                # spikes from recording, dump them...
                last = save_data(f, last, network)

        for j in range(0,  4000):
            network.set_rate([j+1], params.rate)

        for t in range(POST_STIM):
            nest.Simulate(1000)

            # spikes from recording, dump them...
            last = save_data(f, last, network)


    #np.save(name+"/final_connectivity."+str(nest.Rank()), sparse.csr_matrix(current))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        help="experiment name",
        required=True
    )

    parser.add_argument(
        "--teacher-strength",
        type=int,
        help="strength of teacher stimulus in percent",
        default=10
    )

    parser.add_argument(
        "--stimulus-duration",
        type=int,
        help="duration of one stimulus",
        default=1
    )

    args = parser.parse_args()

    # global settings
    print(args.teacher_strength)

    #params.slope = 0.8

    simulation(args.name, args.teacher_strength, args.stimulus_duration)
    #print(args.name, args.with_teacher, args.with_plasticity)