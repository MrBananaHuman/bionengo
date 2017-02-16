import nengo
import numpy as np
import ipdb
import json
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
from BahlNeuronSystem import BahlNeuron, post_build_func
from CustomSolverSystem import CustomSolver
from optimize_bioneuron_system_onebyone import make_signal,ch_dir


def main():
	P=eval(open('parameters.txt').read())
	if 'directory' in P:
		P['load_weights']=True
		os.chdir(P['directory'])
	else:
		P['load_weights']=False
		datadir=ch_dir()
		P['directory']=datadir

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

		ens_bio3=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P),label='ens_bio3',										
								seed=P['ens_ideal3_seed'],radius=P['radius_ideal'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		ens_lif3=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_ideal3_seed'],
								radius=P['radius_ideal'],label='ens_lif3',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		ens_dir3=nengo.Ensemble(n_neurons=1,dimensions=P['dim'],
								neuron_type=nengo.Direct(),label='ens_dir3')

		ens_bio4=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P),label='ens_bio4',										
								seed=P['ens_ideal4_seed'],radius=P['radius_ideal'],
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		ens_lif4=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),seed=P['ens_ideal4_seed'],
								radius=P['radius_ideal'],label='ens_lif4',
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],P['max_ideal_rate']))
		ens_dir4=nengo.Ensemble(n_neurons=1,dimensions=P['dim'],
								neuron_type=nengo.Direct(),label='ens_dir4')


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
		nengo.Connection(pre3,ens_bio3,
							synapse=P['tau'],
							transform=P['transform_pre3_to_ens3'])
		nengo.Connection(pre3,ens_lif3,
							synapse=P['tau'],
							transform=P['transform_pre3_to_ens3'])
		nengo.Connection(stim3,ens_dir3,
							synapse=P['tau'],
							transform=P['transform_pre3_to_ens3'])


		solver_ens_bio=CustomSolver(P,ens_bio,model,method=P['decoder_train'])
		solver_ens_lif=nengo.solvers.LstsqL2()

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
		solver_ens_lif2=nengo.solvers.LstsqL2()

		nengo.Connection(ens_bio2,ens_bio3,
							synapse=P['tau'],
							solver=solver_ens_bio2,
							transform=P['transform_ens2_to_ens3'])
		nengo.Connection(ens_lif2,ens_lif3,
							synapse=P['tau'],
							solver=solver_ens_lif2,
							transform=P['transform_ens2_to_ens3'])
		nengo.Connection(ens_dir2,ens_dir3,
							synapse=P['tau'],
							transform=P['transform_ens2_to_ens3'])

		solver_ens_bio3=CustomSolver(P,ens_bio3,model,method=P['decoder_train'])
		solver_ens_lif3=nengo.solvers.LstsqL2()

		nengo.Connection(ens_bio3,ens_bio4,
							synapse=P['tau'],
							solver=solver_ens_bio3,
							transform=P['transform_ens3_to_ens4'])
		nengo.Connection(ens_lif3,ens_lif4,
							synapse=P['tau'],
							solver=solver_ens_lif3,
							transform=P['transform_ens3_to_ens4'])
		nengo.Connection(ens_dir3,ens_dir4,
							synapse=P['tau'],
							transform=P['transform_ens3_to_ens4'])

		solver_ens_bio4=CustomSolver(P,ens_bio4,model,method=P['decoder_train'])
		solver_ens_lif4=nengo.solvers.LstsqL2()


		'''PROBES'''
		probe_stim=nengo.Probe(stim,synapse=None)
		probe_pre=nengo.Probe(pre,synapse=P['kernel']['tau'])
		probe_stim2=nengo.Probe(stim2,synapse=None)
		probe_pre2=nengo.Probe(pre2,synapse=P['kernel']['tau'])
		probe_stim3=nengo.Probe(stim3,synapse=None)
		probe_pre3=nengo.Probe(pre3,synapse=P['kernel']['tau'])

		probe_bio_spikes=nengo.Probe(ens_bio.neurons,'spikes')
		probe_lif_spikes=nengo.Probe(ens_lif.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio,synapse=P['kernel']['tau'],solver=solver_ens_bio)
		probe_lif=nengo.Probe(ens_lif,synapse=P['kernel']['tau'],solver=solver_ens_lif)
		probe_dir=nengo.Probe(ens_dir,synapse=P['kernel']['tau'])

		probe_bio_spikes2=nengo.Probe(ens_bio2.neurons,'spikes')
		probe_lif_spikes2=nengo.Probe(ens_lif2.neurons,'spikes')
		probe_bio2=nengo.Probe(ens_bio2,synapse=P['kernel']['tau'],solver=solver_ens_bio2)
		probe_lif2=nengo.Probe(ens_lif2,synapse=P['kernel']['tau'],solver=solver_ens_lif2)
		probe_dir2=nengo.Probe(ens_dir2,synapse=P['kernel']['tau'])

		probe_bio_spikes3=nengo.Probe(ens_bio3.neurons,'spikes')
		probe_lif_spikes3=nengo.Probe(ens_lif3.neurons,'spikes')
		probe_bio3=nengo.Probe(ens_bio3,synapse=P['kernel']['tau'],solver=solver_ens_bio3)
		probe_lif3=nengo.Probe(ens_lif3,synapse=P['kernel']['tau'],solver=solver_ens_lif3)
		probe_dir3=nengo.Probe(ens_dir3,synapse=P['kernel']['tau'])

		probe_bio_spikes4=nengo.Probe(ens_bio4.neurons,'spikes')
		probe_lif_spikes4=nengo.Probe(ens_lif4.neurons,'spikes')
		probe_bio4=nengo.Probe(ens_bio4,synapse=P['kernel']['tau'],solver=solver_ens_bio4)
		probe_lif4=nengo.Probe(ens_lif4,synapse=P['kernel']['tau'],solver=solver_ens_lif4)
		probe_dir4=nengo.Probe(ens_dir4,synapse=P['kernel']['tau'])

	with nengo.Simulator(model,post_build_func=post_build_func,dt=P['dt_nengo']) as sim:
		sim.run(P['test']['t_final'])
	
	sns.set(context='poster')
	os.chdir(P['directory'])
	figure1, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,sharex=True)
	ax1.plot(sim.trange(),sim.data[probe_stim],label='stim')
	ax1.plot(sim.trange(),sim.data[probe_stim2],label='stim2')
	ax1.plot(sim.trange(),sim.data[probe_stim3],label='stim3')
	ax2.plot(sim.trange(),sim.data[probe_pre],label='pre_to_ens_1')
	ax2.plot(sim.trange(),sim.data[probe_pre2],label='pre2_to_ens_1')
	ax2.plot(sim.trange(),sim.data[probe_pre3],label='pre3_to_ens_2')
	legend1=ax1.legend(prop={'size':8})
	legend2=ax2.legend(prop={'size':8})
	ax1.set(ylabel='$x(t)$') #,ylim=((np.min(raw_signal),np.max(raw_signal)))
	ymin=ax1.get_ylim()[0]
	ymax=ax1.get_ylim()[1]	
	ax2.set(ylabel='pre (lif)')#,ylim=((ymin,ymax)))

	rmse_bio=np.sqrt(np.average((sim.data[probe_dir]-sim.data[probe_bio])**2))
	rmse_lif=np.sqrt(np.average((sim.data[probe_dir]-sim.data[probe_lif])**2))
	ax3.plot(sim.trange(),sim.data[probe_bio],label='bio, rmse=%.5f'%rmse_bio)
	ax3.plot(sim.trange(),sim.data[probe_lif],label='lif, rmse=%.5f'%rmse_lif)
	ax3.plot(sim.trange(),sim.data[probe_dir],label='direct')
	ax3.set(ylabel='ens_1 $\hat{x}(t)$')#,ylim=((ymin,ymax)))
	legend3=ax3.legend(prop={'size':8})

	rmse_bio2=np.sqrt(np.average((sim.data[probe_dir2]-sim.data[probe_bio2])**2))
	rmse_lif2=np.sqrt(np.average((sim.data[probe_dir2]-sim.data[probe_lif2])**2))
	ax4.plot(sim.trange(),sim.data[probe_bio2],label='bio, rmse=%.5f'%rmse_bio2)
	ax4.plot(sim.trange(),sim.data[probe_lif2],label='lif, rmse=%.5f'%rmse_lif2)
	ax4.plot(sim.trange(),sim.data[probe_dir2],label='direct')
	ax4.set(ylabel='ens_2 $\hat{x}(t)$')#,ylim=((ymin,ymax)))
	legend4=ax4.legend(prop={'size':8})

	rmse_bio3=np.sqrt(np.average((sim.data[probe_dir3]-sim.data[probe_bio3])**2))
	rmse_lif3=np.sqrt(np.average((sim.data[probe_dir3]-sim.data[probe_lif3])**2))
	ax5.plot(sim.trange(),sim.data[probe_bio3],label='bio, rmse=%.5f'%rmse_bio3)
	ax5.plot(sim.trange(),sim.data[probe_lif3],label='lif, rmse=%.5f'%rmse_lif3)
	ax5.plot(sim.trange(),sim.data[probe_dir3],label='direct')
	ax5.set(ylabel='ens_3 $\hat{x}(t)$')#,ylim=((ymin,ymax)))
	legend5=ax5.legend(prop={'size':8})

	rmse_bio4=np.sqrt(np.average((sim.data[probe_dir4]-sim.data[probe_bio4])**2))
	rmse_lif4=np.sqrt(np.average((sim.data[probe_dir4]-sim.data[probe_lif4])**2))
	ax6.plot(sim.trange(),sim.data[probe_bio4],label='bio, rmse=%.5f'%rmse_bio4)
	ax6.plot(sim.trange(),sim.data[probe_lif4],label='lif, rmse=%.5f'%rmse_lif4)
	ax6.plot(sim.trange(),sim.data[probe_dir4],label='direct')
	ax6.set(ylabel='ens_4 $\hat{x}(t)$')#,ylim=((ymin,ymax)))
	legend6=ax6.legend(prop={'size':8})

	figure1.savefig('bioneuron_vs_LIF_decode.png')


	bio_rates=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_bio_spikes][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_bio_spikes].shape[1])]).T
	lif_rates=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_lif_spikes][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_lif_spikes].shape[1])]).T
	rmse_rates=np.sqrt(np.average((bio_rates-lif_rates)**2))
	
	bio_rates2=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_bio_spikes2][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_bio_spikes2].shape[1])]).T
	lif_rates2=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_lif_spikes2][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_bio_spikes2].shape[1])]).T
	rmse_rates2=np.sqrt(np.average((bio_rates2-lif_rates2)**2))
	
	bio_rates3=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_bio_spikes3][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_bio_spikes3].shape[1])]).T
	lif_rates3=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_lif_spikes3][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_bio_spikes3].shape[1])]).T
	rmse_rates3=np.sqrt(np.average((bio_rates3-lif_rates3)**2))
	
	bio_rates4=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_bio_spikes4][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_bio_spikes4].shape[1])]).T
	lif_rates4=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_lif_spikes4][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_bio_spikes4].shape[1])]).T
	rmse_rates4=np.sqrt(np.average((bio_rates4-lif_rates4)**2))

	figure2, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex=True)

	rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax1,use_eventplot=True)
	ax1.set(ylabel='bioneuron',yticks=([]),title='rmse (rates) = %.5f'%rmse_rates)
	rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax2,use_eventplot=True)
	ax2.set(ylabel='lif',yticks=([]))

	rasterplot(sim.trange(),sim.data[probe_bio_spikes2],ax=ax3,use_eventplot=True)
	ax3.set(ylabel='bioneuron2',yticks=([]),title='rmse (rates) = %.5f'%rmse_rates2)
	rasterplot(sim.trange(),sim.data[probe_lif_spikes2],ax=ax4,use_eventplot=True)
	ax4.set(ylabel='lif2',yticks=([]))

	rasterplot(sim.trange(),sim.data[probe_bio_spikes3],ax=ax5,use_eventplot=True)
	ax5.set(ylabel='bioneuron3',yticks=([]),title='rmse (rates) = %.5f'%rmse_rates3)
	rasterplot(sim.trange(),sim.data[probe_lif_spikes3],ax=ax6,use_eventplot=True)
	ax6.set(ylabel='lif3',yticks=([]))

	rasterplot(sim.trange(),sim.data[probe_bio_spikes4],ax=ax7,use_eventplot=True)
	ax7.set(ylabel='bioneuron4',xlabel='time (s)',yticks=([]),title='rmse (rates) = %.5f'%rmse_rates4)
	rasterplot(sim.trange(),sim.data[probe_lif_spikes4],ax=ax8,use_eventplot=True)
	ax8.set(ylabel='lif4',xlabel='time (s)',yticks=([]))

	figure2.savefig('bioneuron_vs_LIF_activity.png')

if __name__=='__main__':
	main()