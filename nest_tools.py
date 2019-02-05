import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby

import nest
import params



class Network:
    spike_detectors = {}

    def __init__(self, plasticity=False, target_rate=17/1000):
        self.plasticity = plasticity
        self.target_rate = target_rate

    def reset_nest(self, print_time=False):
        nest.ResetKernel()
        nest.set_verbosity("M_FATAL")

        if self.plasticity:
            nest.EnableStructuralPlasticity()

        nest.SetKernelStatus({
            'resolution': params.dt,
            'print_time': print_time,
            'structural_plasticity_update_interval': int(params.msp_update_interval / params.dt),  # update interval for MSP in time steps
            'local_num_threads': 1,
            'grng_seed': params.grng_seed,
            'rng_seeds': range(params.seed+1,params.seed+1 + nest.NumProcesses()),
            'data_path': 'data',
            "overwrite_files": True
        })

        

    def setup_static_network(self):
        nest.SetDefaults(params.neuron_model, params.neuron_params)
        nest.CopyModel(params.neuron_model, 'excitatory')
        nest.CopyModel(params.neuron_model, 'inhibitory')
        
        # types of synapses
        nest.CopyModel("static_synapse", "inhibitory_synapse", {"weight":-params.g*params.J, "delay":params.delay})
        nest.CopyModel("static_synapse", "excitatory_synapse", {"weight":params.J, "delay":params.delay})
        nest.CopyModel("static_synapse", "plastic_synapse", {"weight":params.J, "delay":params.delay})
        nest.SetDefaults('plastic_synapse',{'weight': params.J,'delay': params.delay})

        if self.plasticity:
            gc_den  = {'growth_curve': params.growth_curve, 'z': params.z0, 'growth_rate': params.slope*self.target_rate, 'eps': self.target_rate, 'continuous': False}
            gc_axon = {'growth_curve': params.growth_curve, 'z': params.z0, 'growth_rate': params.slope*self.target_rate, 'eps': self.target_rate, 'continuous': False}
            nest.SetDefaults('excitatory', 'synaptic_elements', {'Axon_exc': gc_axon, 'Den_exc': gc_den})


            # Use SetKernelStatus to activate the plastic synapses
            nest.SetKernelStatus({
                'structural_plasticity_synapses': {
                    'syn1': {
                        'model': 'plastic_synapse',
                        'post_synaptic_element': 'Den_exc',
                        'pre_synaptic_element': 'Axon_exc',
                    }
                },
                'autapses': False,
            })

        # create populations
        self.excitatory_neurons = nest.Create('excitatory', params.NE)
        self.inhibitory_neurons = nest.Create('inhibitory', params.NI)

        if self.plasticity:
            nest.Connect(
                self.excitatory_neurons, 
                self.inhibitory_neurons,
                {'rule': 'fixed_indegree','indegree': params.CE},
                'excitatory_synapse'
            )
            
            nest.Connect(
                self.inhibitory_neurons, 
                self.excitatory_neurons+self.inhibitory_neurons,
                {'rule': 'fixed_indegree','indegree': params.CI},
                'inhibitory_synapse'
            )

        else:

            # connect populations

            # 10% connectivity from excitatory population to all other neurons
            nest.Connect(
                self.excitatory_neurons,                          # from excitatory neurons
                self.excitatory_neurons+self.inhibitory_neurons,  # to all neurons
                {'rule': 'fixed_indegree','indegree': params.CE}, # fixed 10% indegree
                'excitatory_synapse'                              # standard exc. synapse
            )

            # 10% connectivity from inhibitory population to all other neurons
            nest.Connect(
                self.inhibitory_neurons,                          # from excitatory neurons
                self.excitatory_neurons+self.inhibitory_neurons,  # to all neurons
                {'rule': 'fixed_indegree','indegree': params.CI}, # fixed 10% indegree
                'inhibitory_synapse'                              # standard inh. synapse - stronger!
            )

        # background input - simple version, one poisson for all
        #poisson_generator = nest.Create('poisson_generator',params={'rate':params.rate})
        #nest.Connect(poisson_generator, self.excitatory_neurons + self.inhibitory_neurons,'all_to_all','excitatory_synapse')

        # background input - complex version, every excitatory neuron controllable
        self.poisson_generator_ex  = nest.Create('poisson_generator', params.NE)
        poisson_generator_inh = nest.Create('poisson_generator')
        nest.SetStatus(self.poisson_generator_ex, {"rate": params.rate})
        nest.SetStatus(poisson_generator_inh, {"rate": params.rate})

        nest.Connect(
            self.poisson_generator_ex,
            self.excitatory_neurons,
            'one_to_one',
            'excitatory_synapse'
        )

        nest.Connect(
            poisson_generator_inh,
            self.inhibitory_neurons,
            'all_to_all',
            'excitatory_synapse'
        )

    def set_rate(self, neurons, rate):
        # calculate positions of generators for neurons
        positions = np.array(neurons, int) - 1

        # select these positions and convert to tuple
        generator_ids = tuple(np.array(self.poisson_generator_ex)[positions])

        # set rate for generators
        nest.SetStatus(generator_ids, {"rate": rate})

    def record_spikes(self, name, start=0.0, end=None, neurons=None):
        detector_params = {'start':start}
        if end: detector_params['end'] = end

        spike_detector = nest.Create('spike_detector',params=detector_params)

        if neurons:
            nest.Connect(neurons, spike_detector,'all_to_all')
        else:
            nest.Connect(self.excitatory_neurons + self.inhibitory_neurons, spike_detector,'all_to_all')

        self.spike_detectors[name] = spike_detector

    def record_spikes_to_file(self):
        detector_params = {
            'to_file': True,
            'to_memory': False,
            "withgid": True, 
            "withtime": True
        }

        spike_detector = nest.Create('spike_detector',params=detector_params)
        nest.Connect(self.excitatory_neurons + self.inhibitory_neurons, spike_detector,'all_to_all')


    def save_recording(self, name, filename):
        detector = self.spike_detectors[name]
        events = nest.GetStatus(detector,'events')[0]
        times = events['times']
        senders = events['senders']

        data = {'times': times, 'senders': senders }

        filename = filename+'.'+str(nest.Rank())+'.npy'

        np.save(filename, data)
    
    def spikes(self, name):
        detector = self.spike_detectors[name]
        events = nest.GetStatus(detector,'events')[0]
        times = events['times']
        senders = events['senders']

        return times, senders

    def reset_recording(self, name):
        detector = self.spike_detectors[name]
        nest.SetStatus(detector,'n_events',0)

    def snapshot_connectivity_matrix(self):
        local_connections = nest.GetConnections()
        
        matrix = np.zeros((params.NE+params.NI, params.NE+params.NI))
        for conn in local_connections:
            if conn[0] <= 5000 and conn[1] <= 5000:
                matrix[conn[1]-1, conn[0]-1] += 1

        return matrix
    
    #def snapshot_connectivity(self):
    #    local_connections = nest.GetConnections(self.excitatory_neurons, self.excitatory_neurons)
    #    sources = np.array(nest.GetStatus(local_connections, 'source'))
    #    targets = np.array(nest.GetStatus(local_connections, 'target'))
    #
    #    return sources, targets

class SpikeRecording():
    @classmethod
    def from_file(cls, name, cores=None):
        if not cores:
            cores = nest.NumProcesses()

        total_times = []
        total_senders = []
        for rank in range(cores):
            filename = name+'.'+str(rank)+'.npy'
            data = np.load(filename)
            times = data.item().get('times')
            senders = data.item().get('senders')
            total_times.append(times)
            total_senders.append(senders)
        return SpikeRecording(np.concatenate(total_times), np.concatenate(total_senders))

    def __init__(self, times, senders):
        self.times = times
        self.senders = senders

    def result(self, neurons=None, start=None, end=None):
        times = self.times
        senders = self.senders

        # filter for neurons
        if neurons:
            positions = np.argwhere(np.isin(senders, neurons))[:,0]
            times = times[positions]
            senders = senders[positions]

        # filter for start
        if start:
            positions = np.argwhere(times >= start)[:,0]
            times = times[positions]
            senders = senders[positions]

        # filter for end
        if end:
            positions = np.argwhere(times <= end)[:,0]
            times = times[positions]
            senders = senders[positions]

        return times, senders

    def spikes_for_neuron(self, neuron, start=None, end=None):
        times, senders = self.result([neuron], start, end)
        return times

    def rate(self, neurons, start, end):
        times, senders = self.result(neurons, start, end)

        rates = np.zeros(len(neurons))

        for index, neuron in enumerate(neurons):
            rates[index] = len(senders[senders==neuron])/(end-start)*1000.0

        return rates

    def plot(self, neurons=None,start=None, end=None):
        times, senders = self.result(neurons, start, end)
        plt.plot(times, senders, '.', markersize=1)
        plt.xlabel("time (ms)")
        plt.ylabel("neuron")



