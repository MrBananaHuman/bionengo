from collections import namedtuple
import neuron

class ExpSyn():

    def __init__(self, sec, weight, tau, e_exc=0.0, e_inh=-80.0):
        self.type = 'ExpSyn'
        self.tau = tau
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.ExpSyn(sec)
        self.syn.tau=1000*self.tau
        self.weight = weight
        if self.weight >= 0.0: self.syn.e = self.e_exc
        else: self.syn.e = self.e_inh
        self.conn = neuron.h.NetCon(None, self.syn) #time of spike arrival assigned in run.py
        self.conn.weight[0]=abs(self.weight)

class Exp2Syn():

    def __init__(self, sec, weight, tau_rise=0.0001, tau_fall=0.01, e_exc=0.0, e_inh=-80.0):
        self.type = 'Exp2Syn'
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.Exp2Syn(sec)
        self.syn.tau1=1000*self.tau_rise
        self.syn.tau2=1000*self.tau_fall
        self.weight = weight
        if self.weight >= 0.0: self.syn.e = self.e_exc
        else: self.syn.e = self.e_inh
        self.conn = neuron.h.NetCon(None, self.syn) #time of spike arrival assigned in run.py
        self.conn.weight[0]=abs(self.weight)

#TODO
# class AlphaSyn():

#     def __init__(self, sec, weight, tau, e_exc=0.0, e_inh=-80.0):
#         self.type = 'AlphaSyn'
#         neuron.h.load_file('/home/pduggins/bionengo/alphasyn.hoc')

#         self.tau = tau
#         self.e_exc = e_exc
#         self.e_inh = e_inh
#         self.syn = neuron.h.AlphaSyn(sec)
#         self.syn.tau=1000*self.tau
#         self.weight = weight
#         if self.weight >= 0.0: self.syn.e = self.e_exc
#         else: self.syn.e = self.e_inh
#         self.conn = neuron.h.NetCon(None, self.syn) #time of spike arrival assigned in run.py
#         self.conn.weight[0]=abs(self.weight)