import nengo
import numpy as np
from BahlNeuron import BahlNeuron
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb

def signal(t):
	return np.sin(t*2*np.pi/(1.0/6))

def main_alt():
	dt_nengo=0.001
	n_in=50
	n_bio=10
	t_sim=1.0
	with nengo.Network() as model:
		stim=nengo.Node(output=lambda t: signal(t))
		ens_in=nengo.Ensemble(n_neurons=n_in,dimensions=1,seed=333)
		test_lif=nengo.Ensemble(n_neurons=n_bio,dimensions=1,
								neuron_type=nengo.LIF())
		ens_out=nengo.Ensemble(n_neurons=n_in,dimensions=1)
		nengo.Connection(stim,ens_in,synapse=None)
		nengo.Connection(ens_in,test_lif)
		probe_in=nengo.Probe(ens_in.neurons,'spikes')
		probe_test=nengo.Probe(test_lif)
		probe_out=nengo.Probe(ens_out)
	with nengo.Simulator(model,dt=dt_nengo) as sim:
		sim.run(t_sim)
	sns.set(context='poster')
	figure1, (ax1,ax4) = plt.subplots(2,1,sharex=True)
	rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
	ax1.set(ylabel='input \nspikes')
	ax4.plot(sim.trange(),signal(sim.trange()),label='$x(t)$') #color='k',ls='-',
	ax4.plot(sim.trange(),sim.data[probe_test],label='LIF $\hat{x}(t)$')
	ax4.set(xlabel='time (s)', ylabel='decoded \nvalue')
	plt.legend(loc='lower left',prop={'size':11})
	plt.show()

def main():
	dt_nengo=0.001
	dt_neuron=0.0001
	filenames=None #TODO: bug when t_sim > t_sample and filenames=None
	# filenames='/home/pduggins/bionengo/'+'data/QMEPDX446/'+'filenames.txt'
	filenames='/home/pduggins/bionengo/'+'data/UJRQGR5ZS/'+'filenames.txt' #with gain, bias
	n_in=50
	n_bio=10
	n_syn=5
	evals=3000
	t_sim=1.0
	kernel_type='gaussian'
	tau_filter=0.01 #0.01, 0.2

	with nengo.Network() as model:
		stim=nengo.Node(output=lambda t: signal(t))
		ens_in=nengo.Ensemble(n_neurons=n_in,dimensions=1,seed=333)
		ens_bio=nengo.Ensemble(n_neurons=n_bio,dimensions=1,
								neuron_type=BahlNeuron())
		test_lif=nengo.Ensemble(n_neurons=n_bio,dimensions=1,
								neuron_type=nengo.LIF())
		# ens_out=nengo.Ensemble(n_neurons=n_in,dimensions=1)

		nengo.Connection(stim,ens_in,synapse=None)
		nengo.Connection(ens_in,ens_bio,synapse=None)
		nengo.Connection(ens_in,test_lif,synapse=None)
		# nengo.Connection(ens_bio,ens_out)

		probe_in=nengo.Probe(ens_in.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio)
		probe_voltage=nengo.Probe(ens_bio.neurons,'voltage')
		probe_spikes=nengo.Probe(ens_bio.neurons,'spikes')
		probe_test=nengo.Probe(test_lif)
		# probe_out=nengo.Probe(ens_out)

	with nengo.Simulator(model,dt=dt_nengo) as sim:
		sim.run(t_sim)

	sns.set(context='poster')
	figure1, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
	# voltages=np.array([bioneuron.nengo_voltages for bioneuron in bionode.biopop]).T
	# voltages[:,1] = voltages[:,1]- 140 #for plotting
	rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
	ax1.set(ylabel='input \nspikes')
	ax2.plot(sim.trange(),sim.data[probe_voltage])
	ax2.set(ylabel='bioneuron \nvoltage')
	rasterplot(sim.trange(),sim.data[probe_spikes],ax=ax3,use_eventplot=True)
	ax3.set(ylabel='bioneuron \nspikes',yticks=([]))
	ax4.plot(sim.trange(),signal(sim.trange()),label='$x(t)$') #color='k',ls='-',
	ax4.plot(sim.trange(),sim.data[probe_bio],label='bioneuron $\hat{x}(t)$')
	ax4.plot(sim.trange(),sim.data[probe_test],label='LIF $\hat{x}(t)$')
	ax4.set(xlabel='time (s)', ylabel='decoded \nvalue')
	# ax4.set(xlim=((t_sim/2,t_sim)),ylim=((-1,1)))
	plt.legend(loc='lower left',prop={'size':11})
	plt.show()

if __name__=='__main__':
	main()