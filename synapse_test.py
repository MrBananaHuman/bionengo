import numpy as np
import nengo
import neuron
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns
# from synapses import ExpSyn

class ExpSyn():
    def __init__(self, sec, weight, tau, e_exc=0.0, e_inh=-80.0):
        self.tau = tau
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.ExpSyn(sec)
        self.syn.tau=1000*self.tau #arbitrary 2x multiply to offset phase shift in bio decode
        self.weight = weight
        if self.weight >= 0.0: self.syn.e = self.e_exc
        else: self.syn.e = self.e_inh
        self.spike_in = neuron.h.NetCon(None, self.syn) #time of spike arrival assigned in nengo step
        self.spike_in.weight[0]=abs(self.weight)

neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc') #todo: hardcoded path
cell=neuron.h.Bahl()
synapses=[]

bias=0
bias_current = neuron.h.IClamp(cell.soma(0.5))
bias_current.delay = 0
bias_current.dur = 1e9
bias_current.amp = bias

v_record = neuron.h.Vector()
v_record.record(cell.soma(0.5)._ref_v)
t_record = neuron.h.Vector()
t_record.record(neuron.h._ref_t)

section=cell.soma(0.5)
weight=0.001
transform=50
tau=0.1 #s
synapse=ExpSyn(section,weight,tau)
synapses.append(synapse)

t_spike=0.2 #s
t_final=0.8 #s
dt=0.0001 #s
neuron.h.dt=1000*dt
neuron.init()

for time in np.arange(0,1000*t_final,1000*dt):
	if time == 1000*t_spike:
		synapses[0].spike_in.event(1000*t_spike)
	neuron.run(time)

with nengo.Network() as model:
	stim=nengo.Node(lambda t: 1.0*(t==t_spike))
	ens=nengo.Ensemble(1,dimensions=1,gain=[1],bias=[0],encoders=[[1]],
						neuron_type=nengo.LIF(tau_rc=tau))
	nengo.Connection(stim,ens.neurons,transform=transform)
	probe=nengo.Probe(ens.neurons,'voltage')
with nengo.Simulator(model,dt=dt) as sim:
	sim.run(t_final)


sns.set(context='poster')
figure1,ax1=plt.subplots(1,1)
ax1.plot(np.array(t_record),np.array(v_record))
ax1.plot(1000*sim.trange(),50*sim.data[probe]-70) #50=delta V b/w rest, spike; -70=rest
ax1.set(xlabel='time (ms)',ylabel='voltage (mV)',title='tau=%sms'%(1000*tau))
figure1.savefig('synapse_test.png')
