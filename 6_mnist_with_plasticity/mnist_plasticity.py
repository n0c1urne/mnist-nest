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

def simulation(name, teacher, plasticity, no_stimulus=True, teacher_strength=10, stimulus_strength=10):
    if not os.path.exists(name):
        try:
            os.makedirs(name)
        except:
            pass

    # reseed numpy
    np.random.seed(0)

    # load data - some preprocessing is done in the module (reshaping)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_tools.load_mnist_data('../mnist.pkl.gz')

    # create pixel samples
    pixel_samples = mnist_tools.create_samples(2000, sample_size=40)

    if plasticity:
        network = nest_tools.Network(plasticity=True, target_rate=8.0/1000)
    else:
        network = nest_tools.Network()
    
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
    with gzip.open(name+'/snapshots'+str(nest.Rank()), 'wb') as f:
        for t in range(INIT_DURATION):
            print("Timestep", t)
            nest.Simulate(1000)

            # save data for this step
            last = save_data(f, last, network)
        
        for t in range(DIGITS):
            input_rates = mnist_tools.calc_rates(x_train[t:t+1], pixel_samples, standardize_per_digit=True, strength=stimulus_strength) * params.rate
            input_rates = input_rates.squeeze()

            #print(input_rates.shape, np.mean(input_rates))

            for j, rate in enumerate(input_rates):
                if no_stimulus:
                    network.set_rate([j+1], params.rate)
                else:
                    network.set_rate([j+1], rate)
                    

            if teacher:
                teacher_stim_index = 2000+y_train[t]*200

                for j in range(2000,  4000):
                    network.set_rate([j+1], params.rate)

                for j in range(teacher_stim_index,  teacher_stim_index+200):
                    network.set_rate([j+1], (1.0 + teacher_strength/100.0) * params.rate)

            print("Nr.", t, "Digit", y_train[t])
            nest.Simulate(1000)

            # save data for this step
            last = save_data(f, last, network)
        
        # after learning, turn of plasticity and teacher stimulus
        for j in range(4000):
            network.set_rate([j+1], params.rate)

        for i in range(10):
             print("cooldown Nr.", i)
             nest.Simulate(1000)

            # save data for this step
            last = save_data(f, last, network)
        
        for t in range(DIGITS, DIGITS+POST_STIM):
            input_rates = mnist_tools.calc_rates(x_train[t:t+1], pixel_samples, standardize_per_digit=True, strength=stimulus_strength) * params.rate
            input_rates = input_rates.squeeze()

            for j, rate in enumerate(input_rates):
                if no_stimulus:
                    network.set_rate([j+1], params.rate)
                else:
                    network.set_rate([j+1], rate)
                
            print("poststim Nr.", t, "Digit", y_train[t])
            nest.Simulate(1000)

            # save data for this step
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
        "--with-teacher",
        nargs='?',
        help="flag for teacher signal",
        const=True,
        default=False
    )

    parser.add_argument(
        "--with-plasticity",
        nargs='?',
        help="flag for plasticity",
        const=True,
        default=False
    )

    parser.add_argument(
        "--without-stimulus",
        nargs='?',
        help="flag for stimulus",
        const=True,
        default=False
    )
    parser.add_argument(
        "--teacher-strength",
        type=int,
        help="strength of teacher stimulus in percent",
        default=10
    )    
    parser.add_argument(
        "--stimulus-strength",
        type=int,
        help="strength of stimulus in percent",
        default=10
    )    
    
    
    args = parser.parse_args()

    # global settings
    simulation(
        args.name, 
        args.with_teacher, 
        args.with_plasticity, 
        args.without_stimulus, 
        args.teacher_strength, 
        args.stimulus_strength
    )
    