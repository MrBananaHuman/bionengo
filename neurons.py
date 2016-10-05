# from collections import namedtuple
import neuron
import numpy as np
from synapses import ExpSyn, Exp2Syn
import os

class Bahl():
    
    def __init__(self):
        neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
        self.cell = neuron.h.Bahl()
        self.connections = {} #index=input neuron, value=synapses

    def start_recording(self):
        self.v_record = neuron.h.Vector()
        self.v_record.record(self.cell.soma(0.5)._ref_v)
        self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
        self.t_record = neuron.h.Vector()
        self.t_record.record(neuron.h._ref_t)
        self.spikes = neuron.h.Vector()
        self.ap_counter.record(neuron.h.ref(self.spikes))

    def add_connection(self,idx):
        self.connections[idx]=[] #a list of each synapse in this connection 

    def add_synapse(self,idx,syn_type,section,weight,tau,tau2):
        if syn_type == 'ExpSyn':
            self.connections[idx].append(ExpSyn(section,weight,tau))
        elif syn_type == 'Exp2Syn':
            self.connections[idx].append(Exp2Syn(section,weight,tau,tau2))
