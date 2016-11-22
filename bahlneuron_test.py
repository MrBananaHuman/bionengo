import nengo
import numpy as np
from BahlNeuron import BahlNeuron
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb

def signal(t):
	return np.sin(t*2*np.pi/(1.0/6))


def main():
	dt_nengo=0.0001
	dt_neuron=0.0001
	filenames=None #TODO: bug when t_sim > t_sample and filenames=None
	filenames='/home/pduggins/bionengo/'+'data/QMEPDX446/'+'filenames.txt'
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
		test_lif=nengo.Ensemble(n_neurons=n_bio,dimensions=1)
		# test_lif=nengo.Ensemble(n_neurons=2,dimensions=1,
		# 						encoders=[[1],[-1]],
		# 						max_rates=[60,60],
		# 						intercepts=[-0.75,-0.75])
		# ens_out=nengo.Ensemble(n_neurons=n_in,dimensions=1)

		nengo.Connection(stim,ens_in,synapse=None)
		nengo.Connection(ens_in,ens_bio,synapse=None)
		nengo.Connection(ens_in,test_lif,synapse=None)
		# nengo.Connection(bionode,ens_out)

		probe_in=nengo.Probe(ens_in.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio)
		probe_test=nengo.Probe(test_lif.neurons,'spikes')
		# probe_out=nengo.Probe(ens_out)

	with nengo.Simulator(model,dt=dt_nengo) as sim:
		sim.run(t_sim)

if __name__=='__main__':
	main()