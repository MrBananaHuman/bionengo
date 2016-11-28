import nengo
import numpy as np
from BahlNeuron import BahlNeuron, CustomSolver
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb
import json

def signal(t):
	return np.sin(t*2*np.pi/(1.0/6))
	# return t

def rate_decoders_opt(filenames):
	f=open(filenames,'r')
	files=json.load(f)
	A_ideal=[]
	A_actual=[]
	gain_ideal=[]
	bias_ideal=[]
	x_sample=[]
	for bio_idx in range(len(files)):
		with open(files[bio_idx],'r') as data_file: 
			bioneuron_info=json.load(data_file)
		A_ideal.append(bioneuron_info['A_ideal'])
		A_actual.append(bioneuron_info['A_actual'])
		gain_ideal.append(bioneuron_info['gain_ideal'])
		bias_ideal.append(bioneuron_info['bias_ideal'])
		x_sample.append(bioneuron_info['x_sample'])
	
	solver=nengo.solvers.LstsqL2()
	decoders,info=solver(np.array(A_actual).T,np.array(x_sample)[0])
	# decoders,info=solver(np.array(A_ideal).T,np.array(x_sample)[0])
	return decoders

def main():
	dt_nengo=0.001
	dt_neuron=0.0001
	filenames=None #TODO: bug when t_sim > t_sample and filenames=None
	# filenames='/home/pduggins/bionengo/'+'data/QMEPDX446/'+'filenames.txt'
	# filenames='/home/pduggins/bionengo/'+'data/UJRQGR5ZS/'+'filenames.txt' #with gain, bias
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
								neuron_type=BahlNeuron(filenames),label='ens_bio')
		test_lif=nengo.Ensemble(n_neurons=n_bio,dimensions=1,
								neuron_type=nengo.LIF(),max_rates=nengo.dists.Uniform(40,60))
		ens_out=nengo.Ensemble(n_neurons=n_in,dimensions=1)
		test_out=nengo.Ensemble(n_neurons=n_in,dimensions=1)

		nengo.Connection(stim,ens_in,synapse=None)
		nengo.Connection(ens_in,ens_bio,synapse=0.01)
		nengo.Connection(ens_in,test_lif,synapse=0.01)
		# nengo.Connection(ens_bio.neurons,ens_out,
		# 						transform=np.ones((1,n_bio))*rate_decoders_opt(filenames),
		# 						synapse=0.01)
		nengo.Connection(ens_bio,ens_out,
								solver=CustomSolver(filenames),
								synapse=0.01)
		nengo.Connection(test_lif,test_out)

		probe_in=nengo.Probe(ens_in.neurons,'spikes')
		probe_voltage=nengo.Probe(ens_bio.neurons,'voltage')
		probe_spikes=nengo.Probe(ens_bio.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio,synapse=0.01)
		probe_test=nengo.Probe(test_lif,synapse=0.01)
		probe_lif_spikes=nengo.Probe(test_lif.neurons,'spikes')
		probe_out=nengo.Probe(ens_out,synapse=0.01)
		probe_test_out=nengo.Probe(test_out,synapse=0.01)

	with nengo.Simulator(model,dt=dt_nengo) as sim:
		sim.run(t_sim)

	sns.set(context='poster')
	figure1, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,sharex=True)
	rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
	ax1.set(ylabel='input \nspikes')
	ax2.plot(sim.trange(),sim.data[probe_voltage])
	ax2.set(ylabel='bioneuron \nvoltage')
	# rasterplot(sim.trange(),sim.data[probe_spikes],ax=ax3,use_eventplot=True)
	# ax3.set(ylabel='bioneuron \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax3,use_eventplot=True)
	ax3.set(ylabel='lif \nspikes',yticks=([]))
	ax4.plot(sim.trange(),signal(sim.trange()),label='$x(t)$') #color='k',ls='-',
	ax4.plot(sim.trange(),sim.data[probe_bio],label='bioneuron $\hat{x}(t)$')
	ax4.plot(sim.trange(),sim.data[probe_test],label='LIF $\hat{x}(t)$')
	ax4.set(ylabel='decoded \nens')
	ax5.plot(sim.trange(),signal(sim.trange()),label='$x(t)$') #color='k',ls='-',
	ax5.plot(sim.trange(),sim.data[probe_out],label='bioneuron $\hat{x}(t)$ ')
	ax5.plot(sim.trange(),sim.data[probe_test_out],label='LIF $\hat{x}(t)$')
	ax5.set(xlabel='time (s)',ylabel='decoded \nens_out')
	plt.legend(loc='lower left')
	figure1.savefig('bioneuron_plots.png')

if __name__=='__main__':
	main()