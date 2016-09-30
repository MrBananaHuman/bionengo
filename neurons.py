# from collections import namedtuple
import neuron
import numpy as np
from synapses import ExpSyn
import os

# class NrnNeuron(NeuronType):
#     """Marks neurons for simulation in Neuron."""

#     def create(self):
#         """Creates the required Neuron objects to simulate the neuron and
#         returns them."""
#         raise NotImplementedError()

#     def _setup_spike_recorder(self, cells):
#         spikes = [neuron.h.Vector() for c in cells]
#         for c, s in zip(cells, spikes):
#             c.out_con.record(neuron.h.ref(s))
#         return spikes

class Bahl():
    
    def __init__(self):
        # super(Bahl, self).__init__()
        neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
        self.cell = neuron.h.Bahl()
        self.synapses = [] 

    def start_recording(self):
        self.v_record = neuron.h.Vector()
        self.v_record.record(self.cell.soma(0.5)._ref_v)
        self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
        self.t_record = neuron.h.Vector()
        self.t_record.record(neuron.h._ref_t)
        self.spikes = neuron.h.Vector()
        self.ap_counter.record(neuron.h.ref(self.spikes))

    def add_synapse(self,section,weight,tau):
        self.synapses.append(ExpSyn(section,weight,tau))

    # def rates_from_current(self, J):
    #     return np.interp(
    #         J, self.rate_table['current'], self.rate_table['rate'])

    # def rates(self, x, gain, bias):
    #     J = gain * x + bias
    #     return self.rates_from_current(J)

    # def gain_bias(self, max_rates, intercepts):
    #     intercepts = np.asarray(intercepts)
    #     max_rates = np.minimum(max_rates, self.rate_table['rate'].max())

    #     min_j = self.rate_table['current'][np.argmax(
    #         self.rate_table['rate'] > 1)]
    #     max_j = self.rate_table['current'][np.argmax(
    #         np.atleast_2d(self.rate_table['rate']).T >= max_rates, axis=0)]

    #     gain = (min_j - max_j) / (intercepts - 1.0)
    #     bias = min_j - gain * intercepts
    #     return gain, bias

    # def step_math(self, dt, J, spiked, cells, voltage):
    #     for c in cells: c.spikes.resize(0)

    #     neuron.run(neuron.h.t + 1000*dt

    #     spiked[:] = [c.spikes.size() > 0 for c in cells]
    #     spiked /= dt
    #     voltage[:] = [c.neuron.soma.v for c in cells]
