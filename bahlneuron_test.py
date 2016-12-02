import nengo
import numpy as np
from BahlNeuron import BahlNeuron, CustomSolver
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb
import json
from optimize_bioneuron import make_signal
import copy

def signal(t,dim):
	return [np.sin((t+d**2*np.pi/(2.0/6))*2*np.pi/(1.0/6)) for d in range(dim)]

def main():
	P={
		'dt_nengo':0.001,
		'dt_neuron':0.0001,
		'filenames':None, #todo: bug when t_sim > t_sample and filenames=None
		# 'filenames':'/home/pduggins/bionengo/data/UJRQGR5ZS/filenames.txt', #with gain, bias
		# 'filenames':'/home/pduggins/bionengo/data/U41GEJX4F/filenames.txt', #with gain, bias
		# 'filenames':'/home/pduggins/bionengo/data/HYS349HZC/filenames.txt', #2D 50 neurons 3k eval
		'filenames':'/home/pduggins/bionengo/data/R6ZLH1S4Z/filenames.txt', #with gain, bias
		'n_in':50,
		'n_bio':20,
		'n_syn':5,
		'dim':2,
		'ens_in_seed':333,
		'ens_LIF_seed':666,
		'min_ideal_rate':20,
		'max_ideal_rate':40,
		'nengo_synapse':0.05,
		't_sim':3.0,


		'kernel_type':'gaussian',
		'signal': #for optimization and decoder calculation
			{'type':'prime_sinusoids'},
			# {'type':'equalpower','max_freq':10.0,'mean':0.0,'std':1.0},
		'kernel': #for smoothing spikes to calculate loss and A matrix
			#{'type':'exp','tau':0.02},
			{'type':'gauss','sigma':0.02,},

		'evals':100,
		't_train':3.0,
		'synapse_tau':0.01,
		'synapse_type':'ExpSyn',
		'w_0':0.0005,#0.0005
		'bias_min':-3.0,
		'bias_max':3.0,
		'n_seg': 5,
		'n_processes':10,
	}
	P2=copy.copy(P)
	P2['filenames']='/home/pduggins/bionengo/data/R6ZLH1S4Z/data/CISCAQTOA/filenames.txt'

	raw_signal=make_signal(P)

	with nengo.Network() as model:
		stim = nengo.Node(lambda t: raw_signal[:,int(t/P['dt_nengo'])]) #all dim, index at t
		ens_in=nengo.Ensemble(n_neurons=P['n_in'],dimensions=P['dim'],
								seed=P['ens_in_seed'])
		ens_bio=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P),label='ens_bio')
		ens_bio2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P2),label='ens_bio2')										
		ens_lif=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_LIF_seed'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))
		node_bio_out=nengo.Node(None,size_in=P['dim'])
		node_lif_out=nengo.Node(None,size_in=P['dim'])

		nengo.Connection(stim,ens_in,synapse=None)
		conn=nengo.Connection(ens_in,ens_bio,synapse=P['nengo_synapse'])
		nengo.Connection(ens_in,ens_lif,synapse=P['nengo_synapse'])
		conn2=nengo.Connection(ens_bio,ens_bio2,solver=CustomSolver(P,conn),synapse=P['nengo_synapse'])
		# conn2=nengo.Connection(ens_in,ens_bio2,synapse=P['nengo_synapse'])
		# nengo.Connection(ens_bio2,node_bio_out,solver=CustomSolver(P,conn2),synapse=P['nengo_synapse'])
		nengo.Connection(ens_bio,node_bio_out,solver=CustomSolver(P2,conn2),synapse=P['nengo_synapse'])
		nengo.Connection(ens_lif,node_lif_out,synapse=P['nengo_synapse'])

		probe_in=nengo.Probe(ens_in.neurons,'spikes')
		probe_voltage=nengo.Probe(ens_bio.neurons,'voltage')
		probe_voltage2=nengo.Probe(ens_bio2.neurons,'voltage')
		probe_bio_spikes=nengo.Probe(ens_bio.neurons,'spikes')
		probe_bio_spikes2=nengo.Probe(ens_bio2.neurons,'spikes')
		probe_lif_spikes=nengo.Probe(ens_lif.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio,synapse=P['nengo_synapse'],solver=CustomSolver(P,conn))
		probe_bio2=nengo.Probe(ens_bio2,synapse=P['nengo_synapse'],solver=CustomSolver(P,conn2))
		probe_lif=nengo.Probe(ens_lif,synapse=P['nengo_synapse'])
		probe_bio_out=nengo.Probe(node_bio_out,synapse=P['nengo_synapse'])
		probe_lif_out=nengo.Probe(node_lif_out,synapse=P['nengo_synapse'])

	with nengo.Simulator(model,dt=P['dt_nengo']) as sim:
		sim.run(P['t_sim'])


	sns.set(context='poster')
	x_in=raw_signal[:,:len(sim.trange())].T
	xhat_bio_out=sim.data[probe_bio_out]
	xhat_lif_out=sim.data[probe_lif_out]
	figure1, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,sharex=True)
	ax1.plot(sim.trange(),x_in,label='$x(t)$')
	ax1.set(ylabel='$x(t)$')
	ax2.plot(sim.trange(),sim.data[probe_bio])
	ax2.set(ylabel='ens_bio $\hat{x}(t)$')
	ax3.plot(sim.trange(),sim.data[probe_lif])
	ax3.set(ylabel='LIF $\hat{x}(t)$')
	ax4.plot(sim.trange(),sim.data[probe_bio2])
	ax4.set(ylabel='ens_bio2 $\hat{x}(t)$')
	ax5.plot(sim.trange(),xhat_bio_out)
	ax5.set(ylabel='node_bio_out \n$\hat{x}(t)$')
	ax6.plot(sim.trange(),xhat_lif_out)
	ax6.set(xlabel='time (s)',ylabel='node_lif_out \n$\hat{x}(t)$')
	# plt.legend(loc='center right', prop={'size':6}, bbox_to_anchor=(1.1,0.8))
	figure1.savefig('bioneuron_vs_LIF_decode.png')


	from optimize_bioneuron import get_rates
	bio_rates=[]
	bio_rates2=[]
	lif_rates=[]
	for b in range(sim.data[probe_bio_spikes].shape[1]):
		bio_spike, bio_rate=get_rates(P,sim.data[probe_bio_spikes][:,b])
		bio_spike2, bio_rate2=get_rates(P,sim.data[probe_bio_spikes2][:,b])
		lif_spike, lif_rate=get_rates(P,sim.data[probe_lif_spikes][:,b])
		bio_rates.append(bio_rate)
		bio_rates2.append(bio_rate2)
		lif_rates.append(lif_rate)
	bio_rates=np.sum(np.array(bio_rates),axis=0)
	bio_rates2=np.sum(np.array(bio_rates2),axis=0)
	lif_rates=np.sum(np.array(lif_rates),axis=0)
	figure1, ((ax2,ax3,ax4),(ax5,ax6,ax7)) = plt.subplots(2,3,sharex=True)
	# rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
	# ax1.set(ylabel='input \nspikes')
	# ax2.plot(sim.trange(),sim.data[probe_voltage])
	# ax2.set(ylabel='bioneuron \nvoltage')
	rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax2,use_eventplot=True)
	ax2.set(ylabel='bioneuron \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_bio_spikes2],ax=ax3,use_eventplot=True)
	ax3.set(ylabel='bioneuron2 \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax4,use_eventplot=True)
	ax4.set(ylabel='lif spikes',yticks=([]))
	ax5.plot(sim.trange(),bio_rates)
	ax5.set(ylabel='bioneuron \nrates')
	# ax6.plot(sim.trange(),bio_rates2)
	# ax6.set(ylabel='bioneuron2 \nrates')
	ax6.plot(sim.trange(), sim.data[probe_voltage2])
	ax6.set(ylabel='bioneuron2 \nvoltage')
	ax7.plot(sim.trange(),lif_rates)
	ax7.set(ylabel='LIF rates')
	figure1.savefig('bioneuron_vs_LIF_activity.png')

	# figure2, ax2=plt.subplots(1,1)
	# ax2.plot(x_in,x_in)
	# ax2.plot(x_in,xhat_bio_out,label='bioneuron, RMSE=%s'
	# 			%np.sqrt(np.average((x_in-xhat_bio_out)**2)))
	# ax2.plot(x_in,xhat_lif_out,label='LIF, RMSE=%s'
	# 			%np.sqrt(np.average((x_in-xhat_lif_out)**2)))
	# ax2.set(xlabel='$x$',ylabel='$\hat{x}$')
	# plt.legend(loc='lower right')
	# figure2.savefig('rmse.png')

	# ipdb.set_trace()
	# figure3, ax3 = plt.subplots(1, 1)
	# for bio_idx in range(P['n_bio']):
	# 	lifplot=ax3.plot(solver.Y[bio_idx,:-2],solver.A_ideal[bio_idx,:-2],linestyle='--')
	# 	bioplot=ax3.plot(solver.Y[bio_idx,:-2],solver.A_actual[bio_idx,:-2],linestyle='-',
	# 						color=lifplot[0].get_color())
	# ax3.plot(0,0,color='k',linestyle='-',label='bioneuron')
	# ax3.plot(0,0,color='k',linestyle='--',label='LIF')
	# ax3.set(xlabel='x',ylabel='firing rate (Hz)',ylim=(0,60))
	# plt.legend(loc='upper center')
	# figure3.savefig('response_curve_comparison.png')

if __name__=='__main__':
	main()