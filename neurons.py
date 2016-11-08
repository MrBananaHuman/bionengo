# from collections import namedtuple
import neuron
import numpy as np
from synapses import ExpSyn, Exp2Syn
import os

class Bahl():
    
    def __init__(self):
        neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
        self.cell = neuron.h.Bahl()
        self.synapses = {} #index=input neuron, value=synapses
        self.vecstim = {} #index=input neuron, value=VecStim object (input spike times)
        self.netcons = {} #index=input neuron, value=NetCon Object (connection b/w VecStim and Syn)

    def start_recording(self):
        self.v_record = neuron.h.Vector()
        self.v_record.record(self.cell.soma(0.5)._ref_v)
        self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
        self.t_record = neuron.h.Vector()
        self.t_record.record(neuron.h._ref_t)
        self.spikes = neuron.h.Vector()
        self.ap_counter.record(neuron.h.ref(self.spikes))
        self.nengo_spike_times=[]
        self.nengo_voltages=[]

    def add_bias(self,bias):
        self.bias = bias
        self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
        self.bias_current.delay = 0
        self.bias_current.dur = 1e9  # TODO; limits simulation time
        self.bias_current.amp = self.bias

    def add_connection(self,idx):
        self.synapses[idx]=[] #list of each synapse in this connection
        self.vecstim[idx]={'vstim':[],'vtimes':[]} #list of input spike times from this neuron
        self.netcons[idx]=[] #list of netcon objects between input vecstim and synapses for this nrn

    def add_synapse(self,idx,syn_type,section,weight,tau,tau2=0.005):
        if syn_type == 'ExpSyn':
            self.synapses[idx].append(ExpSyn(section,weight,tau))
        elif syn_type == 'Exp2Syn':
            self.synapses[idx].append(Exp2Syn(section,weight,tau,tau2))
