import numpy as np

seed = 549547
procs = 4
grng_seed               = seed
numpy_seed              = seed+1000
np.random.seed(numpy_seed)

# General simulation parameters
dt                  = 0.1   # simulation resolution (ms)
msp_update_interval = 100   # update interval for MSP (ms)


g     = 15.0                # ratio between maximum amplitude of EPSP and EPSP
eta   = 1.5                 # ratio between external rate and external frequency needed for the mean input to reach threshold in absence of feedback
eps   = 0.1                 # connection probability for static connections (all but EE)
order = 1000                # order of network size
NE    = 4*order             # number of excitatory neurons
NI    = 1*order             # number of inhibitory neurons
N     = NE+NI               # total number of neurons
CE    = int(eps*NE)         # number of incoming excitatory synapses per inhibitory neuron
CI    = int(eps*NI)         # number of incominb inhibitory synapses per neuron


neuron_model = "iaf_psc_delta"
CMem         = 250.0                # membrane capacitance (pF)
tauMem       = 20.0                 # membrane time constant (ms)
theta        = 20.0                 # spike threshold (mV)
t_ref        = 2.                   # refractory period (ms)
E_L          = 0.                   # resting membrane potential (mV)
V_reset      = 10.                  # reset potential of the membrane (mV)
V_m          = 0.                   # initial membrane potential (mV)
tau_Ca       = 1000.                # time constant for calcium trace (ms)
beta_Ca      = 1./tau_Ca            # increment on calcium trace per spike (1/ms)
J            = 0.1                  # postsynaptic amplitude in mV
delay        = 1.                   # synaptic delay (ms)

neuron_params   = {
                    "C_m"       : CMem,
                    "tau_m"     : tauMem,
                    "t_ref"     : t_ref,
                    "E_L"       : E_L,
                    "V_reset"   : V_reset,
                    "V_m"       : V_m,
                    "beta_Ca"   : beta_Ca,
                    "tau_Ca"    : tau_Ca,
                    "V_th"      : theta
                   }


# External input rate
nu_th  = theta/(J*CE*tauMem)
nu_ex  = eta*nu_th
rate = 1000.0*nu_ex*CE

print("######### rate", rate)

# Parameter for structural plasticity
growth_curve  = "linear"            # type of growth curve for synaptic elements
z0            = 1.                  # initial number of synaptic elements
slope         = 0.5                 # slope of growth curve for synaptic elements
synapse_model = "static_synapse"    # plastic EE synapse type

#target_rate = 17.1/1000.0
