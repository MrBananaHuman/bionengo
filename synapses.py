from collections import namedtuple
import neuron

class ExpSyn():

    def __init__(self, sec, weight, tau, e_exc=0.0, e_inh=-80.0):
        # super(ExpSyn, self).__init__()
        self.tau = tau
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.ExpSyn(sec)
        self.syn.tau=1000*self.tau
        self.weight = weight
        if self.weight >= 0.0: self.syn.e = self.e_exc
        else: self.syn.e = self.e_inh
        self.conn = neuron.h.NetCon(None, self.syn) #inputs assigned later
        self.conn.weight[0]=abs(self.weight)

class AlphaSyn():

    def __init__(self, sec, weight, tau, e_exc=0.0, e_inh=-80.0):
        # super(AlphaSyn, self).__init__()
        self.tau = tau
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.AlphaSynapse(sec)
        self.syn.tau=1000*self.tau
        self.gmax = weight
        if self.gmax >= 0.0: self.syn.e = self.e_exc
        else: self.syn.e = self.e_inh
        self.conn = neuron.h.NetCon(None, self.syn) #broken
        self.conn.weight[0]=abs(self.gmax)