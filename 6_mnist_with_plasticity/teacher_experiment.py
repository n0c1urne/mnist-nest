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


def simulation(name, teacher_strength):
    if not os.path.exists(name):
        try:
            os.makedirs(name)
        except:
            pass

    # reseed numpy
    np.random.seed(0)
    network = nest_tools.Network(plasticity=True, target_rate=8.0/1000)
    
    network.reset_nest(print_time=False)
    network.setup_static_network()

    # initialize spike detector
    network.record_spikes("recording")

    # connectivity matrix
    last = np.zeros((5000,5000))

    INIT_DURATION = 300
    DIGITS = 1000
    POST_STIM = 1000

    # open a gzipped file to record all data...
    with gzip.open(name+'/snapshots'+str(nest.Rank()), 'wb') as f:
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
            teacher_stim_index = 2000+y_train[t]*200

            for j in range(2000,  4000):
                network.set_rate([j+1], params.rate)

            for j in range(teacher_stim_index,  teacher_stim_index+200):
                network.set_rate([j+1], (1.0 + teacher_strength/100.0) * params.rate)

            print("Nr.", t, "Digit", y_train[t])

            for i in range(10):
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

    np.save(name+"/final_connectivity."+str(nest.Rank()), sparse.csr_matrix(current))

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

    args = parser.parse_args()

    # global settings
    print(args.teacher_strength)

    simulation(args.name, args.teacher_strength)
    #print(args.name, args.with_teacher, args.with_plasticity)