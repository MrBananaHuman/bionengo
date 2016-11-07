import nengo
import numpy as np
from neuron_methods import BioneuronNode
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot

dt=0.0001
with nengo.Network() as model:
	stim=nengo.Node(output=1.0)
	ens=nengo.Ensemble(n_neurons=25,dimensions=1)
	bionode=BioneuronNode(25,5,20)
	nengo.Connection(ens.neurons,bionode)
	probe_in=nengo.Probe(ens.neurons,'spikes')
	probe=nengo.Probe(bionode)

with nengo.Simulator(model,dt=dt) as sim:
	sim.run(0.1)

sns.set(context='poster')
figure, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
voltages=np.array([np.array(bioneuron.v_record) for bioneuron in bionode.biopop])
spike_times=np.array([np.array(bioneuron.spikes) for bioneuron in bionode.biopop])
rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
for vs in range(len(voltages)):
	ax2.plot(sim.trange(),voltages[vs][:len(sim.trange())])
biospikes=[]
for st in range(len(spike_times)):
	spike_train=np.zeros_like(sim.trange())
	for idx in spike_times[st][:len(sim.trange())]/dt/1000:
		if idx >= len(spike_train): break
		spike_train[idx]=1.0/dt
	biospikes.append(spike_train)
biospikes=np.array(biospikes).T
# ipdb.set_trace()
rasterplot(sim.trange(),biospikes,ax=ax3,use_eventplot=True)
plt.show()
