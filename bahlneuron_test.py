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
	raw_signal3=make_signal(P['test3'])

	with nengo.Network(label='test_model') as model:
		'''NODES and ENSEMBLES'''
		stim = nengo.Node(lambda t: raw_signal[:,np.floor(t/P['dt_nengo'])]) #all dim, index at t
		pre=nengo.Ensemble(n_neurons=P['ens_pre_neurons'],dimensions=P['dim'],
								seed=P['ens_pre_seed'],label='pre',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))

		stim2 = nengo.Node(lambda t: raw_signal2[:,int(t/P['dt_nengo'])]) #all dim, index at t
		pre2=nengo.Ensemble(n_neurons=P['ens_pre_neurons'],dimensions=P['dim'],
								seed=P['ens_pre2_seed'],label='pre2',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))

		stim3 = nengo.Node(lambda t: raw_signal3[:,int(t/P['dt_nengo'])]) #all dim, index at t
		pre3=nengo.Ensemble(n_neurons=P['ens_pre_neurons'],dimensions=P['dim'],
								seed=P['ens_pre3_seed'],label='pre3',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))


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

		# ens_bio3=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
		# 						neuron_type=BahlNeuron(P),label='ens_bio3',										
		# 						seed=P['ens_ideal3_seed'],radius=P['radius_ideal'],
		# 						max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		# ens_lif3=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
		# 						neuron_type=nengo.LIF(),seed=P['ens_ideal3_seed'],
		# 						radius=P['radius_ideal'],label='ens_lif3',
		# 						max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		# ens_dir3=nengo.Ensemble(n_neurons=1,dimensions=P['dim'],
		# 						neuron_type=nengo.Direct(),label='ens_dir3')

		# ens_bio4=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
		# 						neuron_type=BahlNeuron(P),label='ens_bio4',										
		# 						seed=P['ens_ideal4_seed'],radius=P['radius_ideal'],
		# 						max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		# ens_lif4=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
		# 						neuron_type=nengo.LIF(),seed=P['ens_ideal4_seed'],
		# 						radius=P['radius_ideal'],label='ens_lif4',
		# 						max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		# ens_dir4=nengo.Ensemble(n_neurons=1,dimensions=P['dim'],
		# 						neuron_type=nengo.Direct(),label='ens_dir4')
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

		nengo.Connection(stim2,pre2,synapse=None)
		nengo.Connection(pre2,ens_bio,
							synapse=P['tau'],
							transform=P['transform_pre2_to_ens'])
		nengo.Connection(pre2,ens_lif,
							synapse=P['tau'],
							transform=P['transform_pre2_to_ens'])
		nengo.Connection(stim2,ens_dir,
							synapse=P['tau'],
							transform=P['transform_pre2_to_ens'])

		nengo.Connection(stim3,pre3,synapse=None)
		nengo.Connection(pre3,ens_bio,
							synapse=P['tau'],
							transform=P['transform_pre3_to_ens'])
		nengo.Connection(pre3,ens_lif,
							synapse=P['tau'],
							transform=P['transform_pre3_to_ens'])
		nengo.Connection(stim3,ens_dir,
							synapse=P['tau'],
							transform=P['transform_pre3_to_ens'])


		solver_ens_bio=CustomSolver(P,ens_bio,model,method=P['decoder_train'])
		solver_ens_lif=CustomSolver(P,ens_lif,model,method='ideal')

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

		solver_ens_bio2=CustomSolver(P,ens_bio2,model,method=P['decoder_train'])
		solver_ens_lif2=CustomSolver(P,ens_lif2,model,method='ideal')


		# nengo.Connection(ens_bio2,ens_bio3,
		# 					synapse=P['tau'],
		# 					solver=solver_ens_bio2,
		# 					transform=P['transform_ens2_to_ens3'])
		# nengo.Connection(ens_lif2,ens_lif3,
		# 					synapse=P['tau'],
		# 					solver=solver_ens_lif2,
		# 					transform=P['transform_ens2_to_ens3'])
		# nengo.Connection(ens_dir2,ens_dir3,
		# 					synapse=P['tau'],
		# 					transform=P['transform_ens2_to_ens3'])

		# solver_ens_bio3=CustomSolver(P,ens_bio2,ens_bio3,model,method=P['decoder_train'])
		# solver_ens_lif3=CustomSolver(P,ens_lif2,ens_lif3,model,method='ideal')

		# nengo.Connection(ens_bio3,ens_bio4,
		# 					synapse=P['tau'],
		# 					solver=solver_ens_bio3,
		# 					transform=P['transform_ens3_to_ens4'])
		# nengo.Connection(ens_lif3,ens_lif4,
		# 					synapse=P['tau'],
		# 					solver=solver_ens_lif3,
		# 					transform=P['transform_ens3_to_ens4'])
		# nengo.Connection(ens_dir3,ens_dir4,
		# 					synapse=P['tau'],
		# 					transform=P['transform_ens3_to_ens4'])

		# solver_ens_bio4=CustomSolver(P,ens_bio3,ens_bio4,model,method=P['decoder_train'])
		# solver_ens_lif4=CustomSolver(P,ens_lif3,ens_lif4,model,method='ideal')


		'''PROBES'''
		probe_stim=nengo.Probe(stim,synapse=None)
		probe_pre=nengo.Probe(pre,synapse=P['tau'])
		probe_stim2=nengo.Probe(stim2,synapse=None)
		probe_pre2=nengo.Probe(pre2,synapse=P['tau'])
		probe_stim3=nengo.Probe(stim3,synapse=None)
		probe_pre3=nengo.Probe(pre3,synapse=P['tau'])

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

		# probe_bio_spikes3=nengo.Probe(ens_bio3.neurons,'spikes')
		# probe_lif_spikes3=nengo.Probe(ens_lif3.neurons,'spikes')
		# probe_bio3=nengo.Probe(ens_bio3,synapse=P['tau'],solver=solver_ens_bio3)
		# probe_lif3=nengo.Probe(ens_lif3,synapse=P['tau'],solver=solver_ens_lif3)
		# probe_dir3=nengo.Probe(ens_dir3,synapse=P['tau'])

		# probe_bio_spikes4=nengo.Probe(ens_bio4.neurons,'spikes')
		# probe_lif_spikes4=nengo.Probe(ens_lif4.neurons,'spikes')
		# probe_bio4=nengo.Probe(ens_bio4,synapse=P['tau'],solver=solver_ens_bio4)
		# probe_lif4=nengo.Probe(ens_lif4,synapse=P['tau'],solver=solver_ens_lif4)
		# probe_dir4=nengo.Probe(ens_dir4,synapse=P['tau'])

	with nengo.Simulator(model,dt=P['dt_nengo']) as sim:
		sim.run(P['test']['t_final'])
	
	sns.set(context='poster')
	
	figure1, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,sharex=True)
	ax1.plot(sim.trange(),sim.data[probe_stim],label='stim')
	ax1.plot(sim.trange(),sim.data[probe_stim2],label='stim2')
	ax1.plot(sim.trange(),sim.data[probe_stim3],label='stim3')
	ax2.plot(sim.trange(),sim.data[probe_pre],label='pre')
	ax2.plot(sim.trange(),sim.data[probe_pre2],label='pre2')
	ax2.plot(sim.trange(),sim.data[probe_pre3],label='pre3')
	legend1=ax1.legend()
	legend2=ax2.legend()
	ax1.set(ylabel='$x(t)$') #,ylim=((np.min(raw_signal),np.max(raw_signal)))
	ymin=ax1.get_ylim()[0]
	ymax=ax1.get_ylim()[1]	
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

	# rmse_bio3=np.sqrt(np.average((sim.data[probe_dir3]-sim.data[probe_bio3])**2))
	# rmse_lif3=np.sqrt(np.average((sim.data[probe_dir3]-sim.data[probe_lif3])**2))
	# ax5.plot(sim.trange(),sim.data[probe_bio3],label='bio, rmse=%.3f'%rmse_bio3)
	# ax5.plot(sim.trange(),sim.data[probe_lif3],label='lif, rmse=%.3f'%rmse_lif3)
	# ax5.plot(sim.trange(),sim.data[probe_dir3],label='direct')
	# ax5.set(ylabel='ens_3 $\hat{x}(t)$',ylim=((ymin,ymax)))
	# legend4=ax5.legend()

	# rmse_bio4=np.sqrt(np.average((sim.data[probe_dir4]-sim.data[probe_bio4])**2))
	# rmse_lif4=np.sqrt(np.average((sim.data[probe_dir4]-sim.data[probe_lif4])**2))
	# ax6.plot(sim.trange(),sim.data[probe_bio4],label='bio, rmse=%.3f'%rmse_bio4)
	# ax6.plot(sim.trange(),sim.data[probe_lif4],label='lif, rmse=%.3f'%rmse_lif4)
	# ax6.plot(sim.trange(),sim.data[probe_dir4],label='direct')
	# ax6.set(ylabel='ens_4 $\hat{x}(t)$',ylim=((ymin,ymax)))
	# legend4=ax6.legend()

	figure1.savefig('bioneuron_vs_LIF_decode.png')


	figure2, ((ax3,ax4),(ax5,ax6)) = plt.subplots(2,2,sharex=True)
	# rasterplot(sim.trange(),sim.data[probe_pre_spikes],ax=ax1,use_eventplot=True)
	# ax1.set(ylabel='input \nspikes',yticks=([]))
	# rasterplot(sim.trange(),sim.data[probe_pre_spikes],ax=ax2,use_eventplot=True)
	# ax2.set(ylabel='input \nspikes',yticks=([]))

	rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax3,use_eventplot=True)
	ax3.set(ylabel='bioneuron \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax4,use_eventplot=True)
	ax4.set(ylabel='lif spikes',yticks=([]))

	rasterplot(sim.trange(),sim.data[probe_bio_spikes2],ax=ax5,use_eventplot=True)
	ax5.set(xlabel='time (s)',ylabel='bioneuron2 \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_lif_spikes2],ax=ax6,use_eventplot=True)
	ax6.set(xlabel='time (s)',ylabel='lif spikes2',yticks=([]))

	figure2.savefig('bioneuron_vs_LIF_activity.png')


	# bio_spikes_nengo=sim.data[probe_bio_spikes]
	# bio_spikes_NEURON=np.array([np.array(nrn.spikes) for nrn in ens_bio.neuron_type.neurons])
	# bio_spikes_train=ens_bio.neuron_type.father_op.inputs['pre']['bio_spikes']	
	# for n in range(P['n_bio']):
	# 	print 'test ens_bio neuron %s nengo spikes:'%n, int(np.sum(bio_spikes_nengo[:,n])*P['dt_nengo'])
	# 	print np.nonzero(bio_spikes_nengo[:,n])
	# 	print 'test ens_bio neuron %s NEURON spikes:'%n, len(bio_spikes_NEURON[n])
	# 	print bio_spikes_NEURON[n]
	# 	print 'test ens_bio neuron %s training spikes:'%n, int(np.sum(bio_spikes_train[n])*P['dt_nengo'])
	# 	print np.nonzero(bio_spikes_train[n])
	# 	print 'test ens_bio neuron %s NEURON voltage:'%n
	# 	print np.array(ens_bio.neuron_type.neurons[n].v_record)

	# bio_spikes_nengo2=sim.data[probe_bio_spikes2]
	# bio_spikes_NEURON2=np.array([np.array(nrn.spikes) for nrn in ens_bio2.neuron_type.neurons])
	# bio_spikes_train2=ens_bio2.neuron_type.father_op.inputs['ens_bio']['bio_spikes']
	# spikes_count=int(np.sum(bio_spikes_nengo)*P['dt_nengo'])
	# spike_times=[]
	# for m in range(sim.data[probe_bio_spikes].shape[1]):
	# 	spike_times.append(np.nonzero(sim.data[probe_bio_spikes][:,m])[0])
	# print 'test ens_bio pre spikes:', spikes_count
	# print spike_times
	# for n in range(P['n_bio']):
	# 	print 'test ens_bio2 neuron %s nengo spikes:'%n, int(np.sum(bio_spikes_nengo2[:,n])*P['dt_nengo'])
	# 	print np.nonzero(bio_spikes_nengo2[:,n])
		# print 'test ens_bio2 neuron %s NEURON spikes:'%n, len(bio_spikes_NEURON2[n])
		# print bio_spikes_NEURON2[n]
		# print 'test ens_bio2 neuron %s weight [0,0]:'%n
		# print ens_bio2.neuron_type.father_op.inputs['ens_bio']['weights'][n][0][0]
		# print 'test ens_bio2 neuron %s training spikes:'%n, int(np.sum(bio_spikes_train2[n])*P['dt_nengo'])
		# print np.nonzero(bio_spikes_train2[n])
		# print 'test ens_bio2 neuron %s NEURON voltage:'%n
		# print np.array(ens_bio2.neuron_type.neurons[n].v_record)

	# from optimize_bioneuron import get_rates
	# import copy
	# P2=copy.copy(P)
	# P2['optimize']['t_final']=P['test']['t_final']
	# # ipdb.set_trace()
	# bio_rates_nengo=np.array([get_rates(P2,bio_spikes_nengo[:,n])[1] for n in range(P['n_bio'])])
	# bio_rates_NEURON=np.array([get_rates(P2,spike_times)[1] for spike_times in bio_spikes_NEURON]).T
	# bio_rates_train=np.array([get_rates(P2,spike_times)[1] for spike_times in bio_spikes_train]).T
	# ideal_rates=np.array([get_rates(P,sim.data[probe_lif_spikes][:,n])[1] for n in range(P['n_bio'])])

	# plots_per_subfig=1
	# n_subfigs=int(np.ceil(P['n_bio']/plots_per_subfig))
	# n_columns = 1
	# n_rows=n_subfigs // n_columns
	# n_rows+=n_subfigs % n_columns
	# position = range(1,n_subfigs + 1)
	# # figure2, axarr= plt.subplots(n_rows,n_columns)
	# k=0
	# for row in range(n_rows):
	# 	for column in range(n_columns):
	# 		figure,ax=plt.subplots(1,1)
	# 		# ax=axarr[row,column]
	# 		bio_rates_temp=[]
	# 		ideal_rates_temp=[]
	# 		for b in range(plots_per_subfig):
	# 			if k>=P['n_bio']: break
	# 			# ipdb.set_trace()
	# 			bio_rates_nengo_plot=ax.plot(sim.trange(),bio_rates_nengo[k],linestyle='-')
	# 			bio_rates_NEURON_plot=ax.plot(sim.trange(),bio_rates_NEURON[:,k],linestyle='-.',
	# 				color=bio_rates_nengo_plot[0].get_color())
	# 			bio_rates_train_plot=ax.plot(sim.trange(),bio_rates_train[:,k],linestyle=':',
	# 				color=bio_rates_nengo_plot[0].get_color())
	# 			ideal_rates_plot=ax.plot(sim.trange(),ideal_rates[k],linestyle='--',
	# 				color=bio_rates_nengo_plot[0].get_color())
	# 			ax.plot(0,0,color='k',linestyle='-',label='nengo')
	# 			ax.plot(0,0,color='k',linestyle='-.',label='NEURON')
	# 			ax.plot(0,0,color='k',linestyle=':',label='train')
	# 			ax.plot(0,0,color='k',linestyle='--',label='LIF')
	# 			ax.legend()
	# 			# bio_rates_temp.append(bio_rates[k])
	# 			# ideal_rates_temp.append(ideal_rates[k])
	# 			k+=1
	# 		# rmse=np.sqrt(np.average((np.array(bio_rates_temp)-np.array(ideal_rates_temp))**2))
	# 		ax.set(xlabel='time (s)',ylabel='firing rate (Hz)')#,title='rmse=%.5f'%rmse)
	# 		figure.savefig('a(t)_bio_vs_ideal_neurons_%s-%s'%(k-plots_per_subfig,k))
	# 		plt.close(figure)

	# ipdb.set_trace()
if __name__=='__main__':
	main()