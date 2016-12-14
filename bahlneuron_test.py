import nengo
import numpy as np
from BahlNeuron import BahlNeuron, CustomSolver
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb
import json
from optimize_bioneuron import make_signal,ch_dir
import copy
import os

def main():
	P={
		'dt_nengo':0.001,
		'dt_neuron':0.0001,
		'inputs':None,
		'directory':None,
		# 'directory':'/home/pduggins/bionengo/data/8CSNR5VL4/',
		# 'directory':'/home/pduggins/bionengo/data/L4EWF1SVW/',
		# 'directory':'/home/pduggins/bionengo/data/CH0HIET0H_goodseed_50pre_50bio_50bio_1000evals/',
		# 'directory':'/home/pduggins/bionengo/data/PKS1UVIH2_2D_150pre_150bio/', #todo: bug when t_sim > t_sample and filenames=None
		# 'directory':'/home/pduggins/bionengo/data/8MMQRNE9R/', #todo: bug when t_sim > t_sample and filenames=None
		# 'directory':'/home/pduggins/bionengo/data/Z0E0G2PN6/', #t2D 40 neurons 2k eval LIF train
		# 'filenames':'/home/pduggins/bionengo/data/HYS349HZC/filenames.txt', #2D 50 neurons 3k eval
		'ens_pre_neurons':30,
		'n_bio':30,
		'n_syn':5,
		'dim':2,
		'ens_pre_seed':222,
		'ens_ideal_seed':333,
		'ens_ideal2_seed':444,
		'min_ideal_rate':80,
		'max_ideal_rate':120,
		'nengo_synapse':0.05,
		't_sim':1.0,

		'kernel_type':'gaussian',
		'signal': #for optimization and decoder calculation
			{'type':'prime_sinusoids'},
			# {'type':'equalpower','max_freq':10.0,'mean':0.0,'std':1.0},
		'kernel': #for smoothing spikes to calculate loss and A matrix
			#{'type':'exp','tau':0.02},
			{'type':'gauss','sigma':0.05,},

		'evals':10,
		't_train':5.0,
		'synapse_tau':0.01,
		'synapse_type':'ExpSyn',
		'w_0':0.001,#0.0005 for n_pre=50 with max_rates=uniform(200,400)
		'bias_min':-2.0,
		'bias_max':2.0,
		'n_seg': 5,
		'n_processes':10,
	}

	if P['directory']==None:
		datadir=ch_dir()
		P['directory']=datadir
	else:
		os.chdir(P['directory'])
	P2=copy.copy(P)
	# P2['signal']={'type':'equalpower','max_freq':10.0,'mean':0.0,'std':1.0} #for signal test != train
	raw_signal=make_signal(P)
	inputs, inputs2 = None, None
	# raw_signal=np.zeros_like(raw_signal) #stimulus off at t=0.5,0.3
	# raw_signal[0,:1500]=0.1*np.ones(1500)
	# raw_signal[1,:2000]=0.2*np.ones(2000)

	with nengo.Network() as model:
		stim = nengo.Node(lambda t: raw_signal[:,int(t/P['dt_nengo'])]) #all dim, index at t
		ens_in=nengo.Ensemble(n_neurons=P['ens_pre_neurons'],dimensions=P['dim'],
								seed=P['ens_pre_seed'],label='pre',
								max_rates=nengo.dists.Uniform(20,30))
		# ens_in2=nengo.Ensemble(n_neurons=P['ens_pre_neurons'],dimensions=P['dim'],
		# 						seed=P['ens_pre_seed'],label='pre2',
		# 						max_rates=nengo.dists.Uniform(10,20))
		# inputs=[
		# 		{'label':ens_in.label,
		# 			'directory':P['directory']+'ens_bio/pre/',
		# 			'filenames':None},
		# 		]
		ens_bio=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P,inputs),label='ens_bio',
								seed=P['ens_ideal_seed'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))
		ens_lif=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_ideal_seed'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))
		# inputs2=[
		# 	{'label':ens_bio.label,
		# 			'directory':P['directory']+'ens_bio2/ens_bio/',
		# 			'filenames':None}
		# 		]
		ens_bio2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P,inputs2),label='ens_bio2',										
								seed=P['ens_ideal2_seed'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))
		ens_lif2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_ideal2_seed'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))				
		node_bio_out=nengo.Node(None,size_in=P['dim'])
		node_lif_out=nengo.Node(None,size_in=P['dim'])

		nengo.Connection(stim,ens_in,synapse=None)
		nengo.Connection(ens_in,ens_bio,synapse=None)
		# nengo.Connection(ens_in2,ens_bio,synapse=P['nengo_synapse'])
		nengo.Connection(ens_in,ens_lif,synapse=P['nengo_synapse'])
		# nengo.Connection(ens_in2,ens_lif,synapse=P['nengo_synapse'])
		# nengo.Connection(ens_bio,ens_bio,solver=CustomSolver(P,ens_bio),synapse=P['nengo_synapse'])
		# nengo.Connection(ens_lif,ens_lif,synapse=P['nengo_synapse'])
		# nengo.Connection(ens_bio,node_bio_out,solver=CustomSolver(P,ens_in,ens_bio),synapse=P['nengo_synapse'])
		# nengo.Connection(ens_lif,node_lif_out,synapse=P['nengo_synapse'])
		nengo.Connection(ens_bio,ens_bio2,solver=CustomSolver(P,ens_in,ens_bio),synapse=P['nengo_synapse'])
		nengo.Connection(ens_lif,ens_lif2,synapse=P['nengo_synapse'])
		nengo.Connection(ens_bio2,node_bio_out,solver=CustomSolver(P,ens_bio,ens_bio2),synapse=P['nengo_synapse'])
		nengo.Connection(ens_lif2,node_lif_out,synapse=P['nengo_synapse'])

		probe_in=nengo.Probe(ens_in,synapse=None)
		probe_in_spikes=nengo.Probe(ens_in.neurons,'spikes')
		# probe_voltage=nengo.Probe(ens_bio.neurons,'voltage')
		# probe_voltage2=nengo.Probe(ens_bio2.neurons,'voltage')
		probe_bio_spikes=nengo.Probe(ens_bio.neurons,'spikes')
		probe_bio_spikes2=nengo.Probe(ens_bio2.neurons,'spikes')
		probe_lif_spikes=nengo.Probe(ens_lif.neurons,'spikes')
		probe_lif_spikes2=nengo.Probe(ens_lif2.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio,synapse=P['nengo_synapse'],
				solver=CustomSolver(P,ens_in,ens_bio))
		probe_bio2=nengo.Probe(ens_bio2,synapse=P['nengo_synapse'],
				solver=CustomSolver(P,ens_bio,ens_bio2))
		probe_lif=nengo.Probe(ens_lif,synapse=P['nengo_synapse'])
		probe_lif2=nengo.Probe(ens_lif2,synapse=P['nengo_synapse'])
		probe_bio_out=nengo.Probe(node_bio_out,synapse=P['nengo_synapse'])
		probe_lif_out=nengo.Probe(node_lif_out,synapse=P['nengo_synapse'])

	with nengo.Simulator(model,dt=P['dt_nengo']) as sim:
		sim.run(P['t_sim'])


	sns.set(context='poster')
	x_in=raw_signal[:,:len(sim.trange())].T
	# figure1, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,sharex=True)
	figure1, ((ax1,ax2),(ax01,ax02),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(5,2,sharex=True)
	ax01.plot(sim.trange(),sim.data[probe_in])
	ax01.set(ylabel='ens_in')
	ax02.plot(sim.trange(),sim.data[probe_in])
	ax02.set(ylabel='ens_in')
	ax1.plot(sim.trange(),x_in)
	ax1.set(ylabel='$x(t)$')
	ax2.plot(sim.trange(),x_in)
	ax2.set(ylabel='$x(t)$')
	ax3.plot(sim.trange(),sim.data[probe_bio])
	ax3.set(ylabel='ens_bio $\hat{x}(t)$')
	ax4.plot(sim.trange(),sim.data[probe_lif])
	ax4.set(ylabel='LIF $\hat{x}(t)$')
	ax5.plot(sim.trange(),sim.data[probe_bio2])
	ax5.set(ylabel='ens_bio2 $\hat{x}(t)$')
	ax6.plot(sim.trange(),sim.data[probe_lif2])
	ax6.set(ylabel='LIF2 $\hat{x}(t)$')
	ax7.plot(sim.trange(),sim.data[probe_bio_out])
	ax7.set(xlabel='time (s)',ylabel='node_bio_out \n$\hat{x}(t)$')
	ax8.plot(sim.trange(),sim.data[probe_lif_out])
	ax8.set(xlabel='time (s)',ylabel='node_lif_out \n$\hat{x}(t)$')
	# plt.legend(loc='center right', prop={'size':6}, bbox_to_anchor=(1.1,0.8))
	figure1.savefig('bioneuron_vs_LIF_decode.png')


	# biodata1=np.load(P['directory']+'ens_bio/'+'biodata.npz')
	# biodata2=np.load(P['directory']+'ens_bio2/'+'biodata.npz')
	# optimization_spikes1=biodata1['bio_spikes'].T
	# optimization_spikes2=biodata2['bio_spikes'].T
	# lif_optimization_spikes1=biodata1['ideal_spikes'].T
	# lif_optimization_spikes2=biodata2['ideal_spikes'].T

	figure2, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,sharex=True)
	rasterplot(sim.trange(),sim.data[probe_in_spikes],ax=ax1,use_eventplot=True)
	ax1.set(ylabel='input \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_in_spikes],ax=ax2,use_eventplot=True)
	ax2.set(ylabel='input \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax4,use_eventplot=True)
	ax4.set(ylabel='lif spikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_lif_spikes2],ax=ax6,use_eventplot=True)
	ax6.set(xlabel='time (s)',ylabel='lif spikes2',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax3,use_eventplot=True)
	ax3.set(ylabel='bioneuron \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_bio_spikes2],ax=ax5,use_eventplot=True)
	ax5.set(xlabel='time (s)',ylabel='bioneuron2 \nspikes',yticks=([]))

	# ax5.plot(sim.trange(),sim.data[probe_voltage])
	# ax5.set(xlabel='time (s)',ylabel='bioneuron \nvoltage')

	# rasterplot(sim.trange(),lif_optimization_spikes1,ax=ax3,use_eventplot=True)
	# ax3.set(ylabel='saved lif_spikes',yticks=([]))
	# rasterplot(sim.trange(),lif_optimization_spikes2,ax=ax5,use_eventplot=True)
	# ax5.set(xlabel='time (s)',ylabel='saved lif_spikes2',yticks=([]))
	# rasterplot(sim.trange(),optimization_spikes1,ax=ax4,use_eventplot=True)
	# ax4.set(ylabel='saved bio_spikes',yticks=([]))
	# rasterplot(sim.trange(),optimization_spikes2,ax=ax6,use_eventplot=True)
	# ax6.set(xlabel='time (s)',ylabel='saved bio_spikes2',yticks=([]))


	figure2.savefig('bioneuron_vs_LIF_activity.png')

if __name__=='__main__':
	main()