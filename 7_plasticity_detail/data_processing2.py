import sys
import os.path
import numpy as np
from math import degrees
import matplotlib.pyplot as plt
from itertools import groupby

sys.path.append('../')
import params
import mnist_tools
import pickle
import gzip  # data loading
from scipy import sparse
from contextlib import ExitStack
import argparse

def process_data(name):
    NE = 5000
    processes = 200
    
    
    # reconstruct final connectivity from changes
    result = np.zeros((NE, NE))
    
    duration = 6000 + 300
    connectivity = np.zeros(duration)
    
    all_times = []
    all_senders = []
    
    #rates = np.zeros((duration, 4000))
    
    #change_matrices = [sparse.csr_matrix((NE, NE)) for i in range(duration)]
    
    matrices = []

    with ExitStack() as stack:
        files = [stack.enter_context(gzip.open(name+'/snapshots'+str(p), 'rb')) for p in range(processes)]
    
        
    
        done = False
        t = 0
        last = None

        with gzip.open('matrix.test', 'wb') as mf:
            while not done:
                print(t)
        
                #if t==100:
                #    done = True

                init = sparse.csr_matrix((4000, 4000), dtype='i1')

                for f in files:
                    try:
                        times = pickle.load(f)
                        senders = pickle.load(f)
                        m = pickle.load(f)
                    
                        init += m[:4000, :4000]
                        all_times.append(times)
                        all_senders.append(senders)
        
                        #for s in senders[senders <= 4000]:
                        #    rates[t, s-1] += 1
        
                    except EOFError:
                        done = True
                
                matrices.append(init)
                t += 1
        
    np.save(name+'_times.npy', np.concatenate(all_times))
    np.save(name+'_senders.npy', np.concatenate(all_senders))
    np.save(name+'_matrices.npy', matrices)
    

    #np.save(name+'_changes.npy', change_matrices)
    #np.save(name+'_rates.npy', rates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        help="experiment name",
        required=True
    )

    args = parser.parse_args()

    process_data(args.name)