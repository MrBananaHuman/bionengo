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
	raw_signal=make_signal(P['test'])
	raw_signal2=make_signal(P['test2'])

	with nengo.Network() as model:
		'''NODES and ENSEMBLES'''
		stim = nengo.Node(lambda t: raw_signal[:,int(t/P['dt_nengo'])]) #all dim, index at t
		pre=nengo.Ensemble(n_neurons=P['ens_pre_neurons'],dimensions=P['dim'],
								seed=P['ens_pre_seed'],label='pre',
								radius=P['radius_ideal'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))

		# stim2 = nengo.Node(lambda t: raw_signal2[:,int(t/P['dt_nengo'])]) #all dim, index at t
		# pre2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
		# 						seed=P['ens_ideal_seed'],label='pre2',
		# 						max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))

		ens_bio=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P),label='ens_bio',
								seed=P['ens_ideal_seed'],radius=P['radius_ideal'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		ens_lif=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_ideal_seed'],
								radius=P['radius_ideal'],label='ens_lif',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		ens_dir=nengo.Ensemble(n_neurons=1,dimensions=P['dim'],
								neuron_type=nengo.Direct(),label='ens_dir')

		ens_bio2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P),label='ens_bio2',										
								seed=P['ens_ideal2_seed'],radius=P['radius_ideal'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		ens_lif2=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_ideal2_seed'],
								radius=P['radius_ideal'],label='ens_lif2',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		ens_dir2=nengo.Ensemble(n_neurons=1,dimensions=P['dim'],
								neuron_type=nengo.Direct(),label='ens_dir2')

		'''CONNECTIONS'''																
		nengo.Connection(stim,pre,synapse=None)
		nengo.Connection(pre,ens_bio,
							synapse=P['tau'],
							transform=P['transform_pre_to_ens'])
		nengo.Connection(pre,ens_lif,
							synapse=P['tau'],
							transform=P['transform_pre_to_ens'])
		nengo.Connection(stim,ens_dir,
							synapse=P['tau'],
							transform=P['transform_pre_to_ens'])

		# nengo.Connection(stim2,pre2,synapse=None)
		# nengo.Connection(pre2,ens_bio,
		# 					synapse=P['tau'],
		# 					transform=P['transform_pre2_to_ens'])
		# nengo.Connection(pre2,ens_lif,
		# 					synapse=P['tau'],
		# 					transform=P['transform_pre2_to_ens'])
		# nengo.Connection(stim2,ens_dir,
		# 					synapse=P['tau'],
		# 					transform=P['transform_pre2_to_ens'])

		solver_ens_bio=CustomSolver(P,pre,ens_bio,model,method=P['rate_decode'])
		solver_ens_lif=CustomSolver(P,pre,ens_lif,model,method='ideal')

		# nengo.Connection(ens_bio,ens_bio,synapse=None,
		# 					transform=P['transform_ens_to_ens'],
		# 					solver=solver_ens_bio)
		# nengo.Connection(ens_lif,ens_lif,
		# 					transform=P['transform_ens_to_ens'],
		# 					synapse=P['tau'],
		# 					solver=solver_ens_lif)

		# nengo.Connection(ens_bio,node_bio_out,solver=solver_ens_bio)
		# nengo.Connection(ens_lif,node_lif_out,synapse=P['tau'])

		nengo.Connection(ens_bio,ens_bio2,
							synapse=P['tau'],
							solver=solver_ens_bio,
							transform=P['transform_ens_to_ens2'])
		nengo.Connection(ens_lif,ens_lif2,
							synapse=P['tau'],
							solver=solver_ens_lif,
							transform=P['transform_ens_to_ens2'])
		nengo.Connection(ens_dir,ens_dir2,
							synapse=P['tau'],
							transform=P['transform_ens_to_ens2'])

		solver_ens_bio2=CustomSolver(P,ens_bio,ens_bio2,model,method=P['rate_decode'])
		solver_ens_lif2=CustomSolver(P,ens_lif,ens_lif2,model,method='ideal')


		'''PROBES'''
		probe_stim=nengo.Probe(stim,synapse=None)
		probe_pre=nengo.Probe(pre,synapse=P['tau'])
		probe_pre_spikes=nengo.Probe(pre.neurons,'spikes')
		
		# probe_stim2=nengo.Probe(stim2,synapse=None)
		# probe_pre2=nengo.Probe(pre2,synapse=P['tau'])
		
		probe_bio_spikes=nengo.Probe(ens_bio.neurons,'spikes')
		probe_lif_spikes=nengo.Probe(ens_lif.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio,synapse=P['tau'],solver=solver_ens_bio)
		probe_lif=nengo.Probe(ens_lif,synapse=P['tau'],solver=solver_ens_lif)
		probe_dir=nengo.Probe(ens_dir,synapse=P['tau'])

		probe_bio_spikes2=nengo.Probe(ens_bio2.neurons,'spikes')
		probe_lif_spikes2=nengo.Probe(ens_lif2.neurons,'spikes')
		probe_bio2=nengo.Probe(ens_bio2,synapse=P['tau'],solver=solver_ens_bio2)
		probe_lif2=nengo.Probe(ens_lif2,synapse=P['tau'],solver=solver_ens_lif2)
		probe_dir2=nengo.Probe(ens_dir2,synapse=P['tau'])

	with nengo.Simulator(model,dt=P['dt_nengo']) as sim:
		sim.run(P['test']['t_final'])
	
	sns.set(context='poster')
	
	figure1, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
	ax1.plot(sim.trange(),sim.data[probe_stim])
	# ax1.plot(sim.trange(),sim.data[probe_stim2])
	ax1.set(ylabel='$x(t)$',ylim=((np.min(raw_signal),np.max(raw_signal))))
	ymin=ax1.get_ylim()[0]
	ymax=ax1.get_ylim()[1]	
	ax2.plot(sim.trange(),sim.data[probe_pre])
	# ax2.plot(sim.trange(),sim.data[probe_pre2])
	ax2.set(ylabel='pre (lif)',ylim=((ymin,ymax)))

	rmse_bio=np.sqrt(np.average((sim.data[probe_dir]-sim.data[probe_bio])**2))
	rmse_lif=np.sqrt(np.average((sim.data[probe_dir]-sim.data[probe_lif])**2))
	ax3.plot(sim.trange(),sim.data[probe_bio],label='bio, rmse=%.3f'%rmse_bio)
	ax3.plot(sim.trange(),sim.data[probe_lif],label='lif, rmse=%.3f'%rmse_lif)
	ax3.plot(sim.trange(),sim.data[probe_dir],label='direct')
	ax3.set(ylabel='ens_1 $\hat{x}(t)$',ylim=((ymin,ymax)))
	legend3=ax3.legend()

	rmse_bio2=np.sqrt(np.average((sim.data[probe_dir2]-sim.data[probe_bio2])**2))
	rmse_lif2=np.sqrt(np.average((sim.data[probe_dir2]-sim.data[probe_lif2])**2))
	ax4.plot(sim.trange(),sim.data[probe_bio2],label='bio, rmse=%.3f'%rmse_bio2)
	ax4.plot(sim.trange(),sim.data[probe_lif2],label='lif, rmse=%.3f'%rmse_lif2)
	ax4.plot(sim.trange(),sim.data[probe_dir2],label='direct')
	ax4.set(ylabel='ens_2 $\hat{x}(t)$',ylim=((ymin,ymax)))
	legend4=ax4.legend()

	figure1.savefig('bioneuron_vs_LIF_decode.png')


	figure2, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,sharex=True)
	rasterplot(sim.trange(),sim.data[probe_pre_spikes],ax=ax1,use_eventplot=True)
	ax1.set(ylabel='input \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_pre_spikes],ax=ax2,use_eventplot=True)
	ax2.set(ylabel='input \nspikes',yticks=([]))

	rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax4,use_eventplot=True)
	ax4.set(ylabel='lif spikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax3,use_eventplot=True)
	ax3.set(ylabel='bioneuron \nspikes',yticks=([]))

	rasterplot(sim.trange(),sim.data[probe_lif_spikes2],ax=ax6,use_eventplot=True)
	ax6.set(xlabel='time (s)',ylabel='lif spikes2',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_bio_spikes2],ax=ax5,use_eventplot=True)
	ax5.set(xlabel='time (s)',ylabel='bioneuron2 \nspikes',yticks=([]))

	figure2.savefig('bioneuron_vs_LIF_activity.png')

	# for nrn in ens_bio.neuron_type.neurons:
	# 	print 'post simulation bias',nrn.bias
	# for nrn2 in ens_bio2.neuron_type.neurons:
	# 	print 'post simulation bias',nrn2.bias

	# lpf=nengo.Lowpass(P['tau'])
	# # ideal=P['transform_pre_to_ens']*\
	# # 	lpf.filt(
	# # 		lpf.filt(
	# # 			raw_signal.T[:len(sim.trange())],
	# # 			dt=P['dt_nengo']),
	# # 		dt=P['dt_nengo'])
	# ideal=P['transform_pre_to_ens']*P['transform_ens_to_ens2']*\
	# 	lpf.filt(
	# 		lpf.filt(
	# 			lpf.filt(
	# 				raw_signal.T[:len(sim.trange())],
	# 				dt=P['dt_nengo']),
	# 			dt=P['dt_nengo']),
	# 		dt=P['dt_nengo'])
	# figure1, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex=True) #,(ax9,ax10)
	# ax1.plot(sim.trange(),sim.data[probe_stim])
	# ax1.set(ylabel='$x(t)$',ylim=((np.min(raw_signal),np.max(raw_signal))))
	# ymin=ax1.get_ylim()[0]
	# ymax=ax1.get_ylim()[1]	
	# ymin_trans=P['transform_pre_to_ens']*ax1.get_ylim()[0]
	# ymax_trans=P['transform_pre_to_ens']*ax1.get_ylim()[1]
	# ymin_trans2=P['transform_pre_to_ens']*P['transform_ens_to_ens2']*ax1.get_ylim()[0]
	# ymax_trans2=P['transform_pre_to_ens']*P['transform_ens_to_ens2']*ax1.get_ylim()[1]
	# ax2.plot(sim.trange(),sim.data[probe_stim])
	# ax2.set(ylabel='$x(t)$',ylim=((ymin,ymax)))
	# ax3.plot(sim.trange(),sim.data[probe_pre])
	# ax3.set(ylabel='pre (lif)',ylim=((ymin,ymax)))
	# ax4.plot(sim.trange(),sim.data[probe_pre])
	# ax4.set(ylabel='pre (lif)',ylim=((ymin,ymax)))
	# ax1.plot(sim.trange(),sim.data[probe_stim2])	
	# ax2.plot(sim.trange(),sim.data[probe_stim2])
	# ax3.plot(sim.trange(),sim.data[probe_in2])
	# ax4.plot(sim.trange(),sim.data[probe_in2])
	# ax5.plot(sim.trange(),sim.data[probe_bio])
	# ax5.set(ylabel='ens_bio $\hat{x}(t)$',ylim=((ymin,ymax)))
	# ax6.plot(sim.trange(),sim.data[probe_lif])
	# ax6.set(ylabel='LIF $\hat{x}(t)$',ylim=((ymin,ymax)))
	# ax7.plot(sim.trange(),sim.data[probe_bio2],label='bio $\hat{x}(t)$')
	# ax7.set(ylabel='ens_bio2 $\hat{x}(t)$',ylim=((ymin,ymax)))
	# # ax7.plot(sim.trange(),sim.data[probe_bio_out],label='bio $\hat{x}(t)$')
	# ax7.plot(sim.trange(),ideal,label='ideal')
	# ax7.set(xlabel='time (s)',ylabel='ens_bio2',ylim=((ymin,ymax)),
	# 	title='\nRMSE=%.3f'%np.sqrt(np.average((ideal-sim.data[probe_bio2])**2)))
	# legend7=ax7.legend(loc='upper left')
	# # ax8.plot(sim.trange(),sim.data[probe_lif_out],label='lif $\hat{x}(t)$')
	# ax8.plot(sim.trange(),sim.data[probe_lif2],label='lif $\hat{x}(t)$')
	# ax8.set(ylabel='LIF2 $\hat{x}(t)$',ylim=((ymin,ymax)))
	# ax8.plot(sim.trange(),ideal,label='ideal')
	# ax8.set(xlabel='time (s)',ylabel='ens_lif2',ylim=((ymin,ymax)),
	# 	title='\nRMSE=%.3f'%np.sqrt(np.average((ideal-sim.data[probe_lif2])**2)))
	# legend8=ax8.legend(loc='upper left')
	#compute numerical sum of raw_signal as the analytic integral to compare with bio/lif integration
	# integral_raw_signal=np.array([np.cumsum(sim.data[probe_in][:,d]*P['dt_nengo'])
	# 								for d in range(sim.data[probe_in].shape[1])]).T
	# ax9.plot(sim.trange(),integral_raw_signal)
	# ax9.set(xlabel='time (s)', ylabel='$\int x(t)$')
	# ax10.plot(sim.trange(),integral_raw_signal)
	# ax10.set(xlabel='time (s)', ylabel='$\int x(t)$')
	# plt.tight_layout()

	# figure1.savefig('bioneuron_vs_LIF_decode_biodecode.png')


if __name__=='__main__':
	main()