import nengo
import numpy as np
from BahlNeuron import BahlNeuron
from CustomSolver import IdentitySolver,CustomSolver
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb
import json
from optimize_bioneuron import make_signal,ch_dir
import copy
import os

def main():
	P=eval(open('parameters.txt').read())
	if P['directory']==None:
		datadir=ch_dir()
		P['directory']=datadir
	else:
		os.chdir(P['directory'])
	with open('parameters.json','w') as fp:
		json.dump(P,fp)
	raw_signal=make_signal(P,'test')
	raw_signal2=make_signal(P,'test')
	# P2=copy.copy(P)
	# P2['signal']={'type':'equalpower','max_freq':5.0,'mean':0.0,'std':0.5} #for signal test != train
	# raw_signal=make_signal(P2)
	raw_signal=np.zeros_like(raw_signal)
	raw_signal[0,:1000]=1.0*np.ones(1000)
	# raw_signal2[0,:1000]=0.5*np.ones(1000)

	with nengo.Network() as model:
		'''nodes and ensembles'''
		stim = nengo.Node(lambda t: raw_signal[:,int(t/P['dt_nengo'])]) #all dim, index at t
		# stim2 = nengo.Node(lambda t: raw_signal2[:,int(t/P['dt_nengo'])]) #all dim, index at t
		ens_in=nengo.Ensemble(n_neurons=P['ens_pre_neurons'],dimensions=P['dim'],
								seed=P['ens_pre_seed'],label='pre',
								radius=P['radius_ideal'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))
		# ens_in2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
		# 						seed=P['ens_ideal_seed'],label='pre2',
		# 						max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
		# 														P['max_ideal_rate']))
		ens_bio=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P),label='ens_bio',
								seed=P['ens_ideal_seed'],radius=P['radius_ideal'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))
		ens_lif=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_ideal_seed'],
								radius=P['radius_ideal'],label='ens_lif',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))
		ens_bio2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P),label='ens_bio2',										
								seed=P['ens_ideal2_seed'],radius=P['radius_ideal'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))
		ens_lif2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_ideal2_seed'],
								radius=P['radius_ideal'],label='ens_lif2',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
																P['max_ideal_rate']))		

		'''connections'''																
		node_bio_out=nengo.Node(None,size_in=P['dim'])
		node_lif_out=nengo.Node(None,size_in=P['dim'])

		nengo.Connection(stim,ens_in,synapse=None)
		# nengo.Connection(stim2,ens_in2,synapse=None)

		P['my_transform']=1.0*P['tau']
		P['my_transform2']=1.0#*P['tau']
		nengo.Connection(ens_in,ens_bio,synapse=None,transform=P['my_transform'])
		nengo.Connection(ens_in,ens_lif,synapse=P['tau'],transform=P['my_transform'])

		# nengo.Connection(ens_in2,ens_bio,synapse=None,transform=P['my_transform2'])
		# nengo.Connection(ens_in2,ens_lif,synapse=P['tau'],transform=P['my_transform2'])

		solver_ens_bio=CustomSolver(P,ens_in,ens_bio,model,recurrent=True)
		solver_ens_lif=CustomSolver(P,ens_in,ens_lif,model,recurrent=True)
		identity_solver=IdentitySolver(P)

		nengo.Connection(ens_bio,ens_bio,solver=solver_ens_bio) #,synapse=P['tau']
		nengo.Connection(ens_lif,ens_lif,synapse=P['tau'],solver=solver_ens_lif)

		# nengo.Connection(ens_bio,node_bio_out,solver=solver_ens_bio) #,synapse=P['tau']
		# nengo.Connection(ens_lif,node_lif_out,synapse=P['tau'])

		nengo.Connection(ens_bio,ens_bio2,synapse=None,solver=identity_solver)
		nengo.Connection(ens_lif,ens_lif2,synapse=P['tau'],solver=solver_ens_lif)

		solver_ens_bio2=CustomSolver(P,ens_bio,ens_bio2,model)
		solver_ens_lif2=CustomSolver(P,ens_lif,ens_lif2,model)

		nengo.Connection(ens_bio2,node_bio_out,solver=solver_ens_bio2,synapse=None)
		nengo.Connection(ens_lif2,node_lif_out,synapse=P['tau'],solver=solver_ens_bio2)

		probe_stim=nengo.Probe(stim,synapse=None)
		# probe_stim2=nengo.Probe(stim2,synapse=None)
		probe_in=nengo.Probe(ens_in,synapse=P['tau'])
		# probe_in2=nengo.Probe(ens_in2,synapse=P['tau'])
		probe_in_spikes=nengo.Probe(ens_in.neurons,'spikes')
		probe_bio_spikes=nengo.Probe(ens_bio.neurons,'spikes')
		probe_bio_spikes2=nengo.Probe(ens_bio2.neurons,'spikes')
		probe_lif_spikes=nengo.Probe(ens_lif.neurons,'spikes')
		probe_lif_spikes2=nengo.Probe(ens_lif2.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio,synapse=P['tau'],solver=solver_ens_bio)
		probe_bio2=nengo.Probe(ens_bio2,synapse=P['tau'],solver=solver_ens_bio2)
		probe_lif=nengo.Probe(ens_lif,synapse=P['tau'],solver=solver_ens_bio) #probe lif
		probe_lif2=nengo.Probe(ens_lif2,synapse=P['tau'],solver=solver_ens_bio2)
		probe_bio_out=nengo.Probe(node_bio_out,synapse=P['tau'])
		probe_lif_out=nengo.Probe(node_lif_out,synapse=P['tau'])

	with nengo.Simulator(model,dt=P['dt_nengo']) as sim:
		sim.run(P['t_test'])

	# ipdb.set_trace()
	sns.set(context='poster')
	# figure1, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,sharex=True)
	figure1, ((ax1,ax2),(ax01,ax02),(ax3,ax4),(ax5,ax6),(ax7,ax8),(ax9,ax10)) = plt.subplots(6,2,sharex=True)
	ax01.plot(sim.trange(),sim.data[probe_in])
	# ax01.plot(sim.trange(),sim.data[probe_in2])
	ax01.set(ylabel='ens_in')
	ax02.plot(sim.trange(),sim.data[probe_in])
	# ax02.plot(sim.trange(),sim.data[probe_in2])
	ax02.set(ylabel='ens_in')
	ax1.plot(sim.trange(),sim.data[probe_stim])
	# ax1.plot(sim.trange(),sim.data[probe_stim2])
	ax1.set(ylabel='$x(t)$')
	ax2.plot(sim.trange(),sim.data[probe_stim])
	# ax2.plot(sim.trange(),sim.data[probe_stim2])
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
	ax7.set(ylabel='node_bio_out \n$\hat{x}(t)$')
		# title='RMSE=%.3f'%np.sqrt(np.average((sim.data[probe_in]-sim.data[probe_bio_out])**2)))
	ax8.plot(sim.trange(),sim.data[probe_lif_out])
	ax8.set(ylabel='node_lif_out \n$\hat{x}(t)$')
		# title='RMSE=%.3f'%np.sqrt(np.average((sim.data[probe_in]-sim.data[probe_lif_out])**2)))
	# plt.legend(loc='center right', prop={'size':6}, bbox_to_anchor=(1.1,0.8))
	#compute numerical sum of raw_signal as the analytic integral to compare with bio/lif integration
	integral_raw_signal=np.array([np.cumsum(sim.data[probe_in][:,d]*P['dt_nengo'])
									for d in range(sim.data[probe_in].shape[1])]).T
	# ipdb.set_trace()
	ax9.plot(sim.trange(),integral_raw_signal)
	ax9.set(xlabel='time (s)', ylabel='$\int x(t)$')
	ax10.plot(sim.trange(),integral_raw_signal)
	ax10.set(xlabel='time (s)', ylabel='$\int x(t)$')
	plt.tight_layout()
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
	# rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax3,use_eventplot=True)
	# ax3.set(ylabel='bioneuron \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_lif_spikes2],ax=ax6,use_eventplot=True)
	ax6.set(xlabel='time (s)',ylabel='lif spikes2',yticks=([]))
	# rasterplot(sim.trange(),sim.data[probe_bio_spikes2],ax=ax5,use_eventplot=True)
	# ax5.set(xlabel='time (s)',ylabel='bioneuron2 \nspikes',yticks=([]))

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