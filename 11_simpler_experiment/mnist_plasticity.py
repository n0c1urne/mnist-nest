import pickle  # da loading
from scipy import sparse
import nest
import sys
import os.path
import numpy as np
from math import degrees
import matplotlib.pyplot as plt
import argparse
import gzip

sys.path.append('../')
import mnist_tools
import nest_tools
import params



def simulate_second(f, last, network):
    """Simulates one second in nest and then stores spike data and changes in connectivity"""
    
    # simulate one second
    nest.Simulate(1000)

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


def simulation(
        name,
        plasticity,
        teacher_strength,
        stimulus_strength,
        stimulus_count,
        stimulus_duration,
        stimulus_pause,
        cooldown):
    """Main function to run simulation"""

    # create output directory, if not exists
    if not os.path.exists(name):
        try:
            os.makedirs(name)
        except:
            pass

    # reseed numpy
    np.random.seed(0)

    # load data - some preprocessing is done in the module (reshaping)
    (x_train, y_train), (x_val, y_val), (x_test,
                                         y_test) = mnist_tools.load_mnist_data('../mnist.pkl.gz')

    # create pixel samples
    pixel_samples = mnist_tools.create_samples(4000, sample_size=40)

    # create default network with or without plasticity
    if plasticity:
        network = nest_tools.Network(plasticity=True, target_rate=8.0/1000)
    else:
        network = nest_tools.Network()

    # initialize network
    network.reset_nest(print_time=False)
    network.setup_static_network()

    # initialize spike detector
    network.record_spikes("recording")

    # connectivity matrix - used to track changes
    last = np.zeros((5000, 5000))

    # time to grow initial network - fixed
    INIT_DURATION = 300

    # open a gzipped file to record all data... different file per process
    with gzip.open(name+'/snapshots'+str(nest.Rank()), 'wb') as f:
        for t in range(INIT_DURATION):
            print("Timestep", t)

            # save data for this step
            last = simulate_second(f, last, network)

        for t in range(stimulus_count):
            print("Stimulus", t, "Digit", y_train[t])

            # ----- DIGIT INPUT -----------

            input_rates = mnist_tools.calc_rates(
                x_train[t:t+1], pixel_samples, standardize_per_digit=True, strength=stimulus_strength)
            input_rates = input_rates.squeeze()
            input_rates[input_rates < 0] = 0  # ensure that no rate is negative

            
            # ----- TEACHER INPUT -----------

            teacher_stim_slice = slice(y_train[t]*400, (y_train[t]+1)*400)

            teacher_rates = np.ones(4000)
            teacher_rates[teacher_stim_slice] = (1.0 + teacher_strength/100.0)

            stim = (input_rates + teacher_rates) / 2  * params.rate
            
            # first reset all teacher rates
            for j in range(0,  4000):
                network.set_rate([j+1], stim[j])

            # ---- STIMULATE ---------------

            for i in range(stimulus_duration):
                # save data for this step
                last = simulate_second(f, last, network)

            # ----- PAUSE ------------------

            # turn of all stimuli for pause
            for j in range(0,  4000):
                network.set_rate([j+1], params.rate)

            for i in range(stimulus_pause):
                # save data for this step
                last = simulate_second(f, last, network)

        for i in range(cooldown):
            print("cooldown Nr.", i)

            # save data for this step
            last = simulate_second(f, last, network)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--stimulus-count",
        type=int,
        help="number of stimuli (first n digits of mnist)",
        required=True
    )

    parser.add_argument(
        "--stimulus-duration",
        type=int,
        help="duration of one stimulus",
        required=True
    )

    parser.add_argument(
        "--stimulus-pause",
        type=int,
        help="duration of pause between stimuli",
        required=True
    )

    parser.add_argument(
        "--with-plasticity",
        nargs='?',
        help="flag for plasticity",
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

    parser.add_argument(
        "--cooldown",
        type=int,
        help="duration of cooldown",
        default=300
    )

    args = parser.parse_args()

    # create unique name from parameters
    name = f"c{args.stimulus_count}_d{args.stimulus_duration}_p{args.stimulus_pause}_s{args.stimulus_strength}_ts{args.teacher_strength}"

    # global settings
    simulation(
        name,
        args.with_plasticity,
        args.teacher_strength,
        args.stimulus_strength,
        args.stimulus_count,
        args.stimulus_duration,
        args.stimulus_pause,
        args.cooldown
    )
