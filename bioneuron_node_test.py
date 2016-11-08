import nengo
import numpy as np
from BioneuronNode import BioneuronNode
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot

def get_spike_train(spike_times):
	biospikes=[]
	for bio_idx in range(len(spike_times)):
		spike_train=np.zeros_like(sim.trange())
		for t_idx in np.array(spike_times[bio_idx])/dt_nengo:
			if t_idx >= len(spike_train): break
			spike_train[t_idx]=1.0/dt_nengo
		biospikes.append(spike_train)
	biospikes=np.array(biospikes).T
	return biospikes

dt_nengo=0.0001
dt_neuron=0.0001
filenames='/home/pduggins/bionengo/'+'data/O8316754B/'+'filenames.txt'
n_in=50
n_bio=5
n_syn=5
evals=10
with nengo.Network() as model:
	stim=nengo.Node(output=1.0)
	ens_in=nengo.Ensemble(n_neurons=n_in,dimensions=1)
	bionode=BioneuronNode(n_in=n_in,n_bio=n_bio,n_syn=n_syn,
							dt_nengo=dt_nengo,dt_neuron=dt_neuron,
							evals=evals,filenames=filenames)
	# ens_out=nengo.Ensemble(n_neurons=n_in,dimensions=1)
	nengo.Connection(ens_in.neurons,bionode)
	# nengo.Connection(bionode,ens_out)
	probe_in=nengo.Probe(ens_in.neurons,'spikes')
	probe_bio=nengo.Probe(bionode)
	# probe_out=nengo.Probe(ens_out)

with nengo.Simulator(model,dt=dt_nengo) as sim:
	sim.run(0.1)

sns.set(context='poster')
figure, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
voltages=np.array([bioneuron.nengo_voltages for bioneuron in bionode.biopop]).T
spike_times=np.array([bioneuron.nengo_spike_times for bioneuron in bionode.biopop]).T
biospikes=get_spike_train(spike_times)
rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
ax1.set(ylabel='lif spikes')
ax2.plot(sim.trange(),voltages)
ax2.set(ylabel='bioneuron voltage \n(dt_nengo)')
rasterplot(sim.trange(),biospikes,ax=ax3,use_eventplot=True)
ax3.set(xlabel='time (s)', ylabel='bioneuron spikes \n(dt_nengo)')
# ax4.plot(sim.trange(),sim.data[probe_out])
# ax4.set(xlabel='time (s)', ylabel='decoded value of ens_out')
plt.show()

