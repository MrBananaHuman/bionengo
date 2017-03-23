import nengo
import numpy as np
import ipdb
import json
import copy
import os
from bioneuron_builder import BahlNeuron, pre_build_func, post_build_func
from bioneuron_solver import BioneuronSolver
from bioneuron_helper import make_signal,ch_dir


def main():
	P=eval(open('bioneuron_parameters.txt').read())
	if 'directory' in P:
		P['load_weights']=True
		os.chdir(P['directory'])
	else:
		P['load_weights']=False
		datadir=ch_dir(P)
		P['directory']=datadir

	with open('parameters.json','w') as fp:
		json.dump(P,fp)
	raw_signal=make_signal(P['test'])

	with nengo.Network(label='test_model') as model:
		'''NODES and ENSEMBLES'''
		stim = nengo.Node(lambda t: raw_signal[:,int(t/P['dt_nengo'])])

		pre=nengo.Ensemble(n_neurons=P['pre']['n_neurons'],dimensions=P['pre']['dim'],
								seed=P['pre']['seed'],label='pre',
								max_rates=nengo.dists.Uniform(P['pre']['min_rate'],P['pre']['max_rate']))
		ens_bio=nengo.Ensemble(n_neurons=P['ens_bio']['n_neurons'],dimensions=P['ens_bio']['dim'],
								neuron_type=BahlNeuron(P,'ens_bio'),label='ens_bio',
								seed=P['ens_bio']['seed'],radius=P['ens_bio']['radius'],
								max_rates=nengo.dists.Uniform(P['ens_bio']['min_rate'],P['ens_bio']['max_rate']))
		ens_lif=nengo.Ensemble(n_neurons=P['ens_bio']['n_neurons'],dimensions=P['ens_bio']['dim'],
								neuron_type=nengo.LIF(),label='ens_lif',
								seed=P['ens_bio']['seed'],radius=P['ens_bio']['radius'],
								max_rates=nengo.dists.Uniform(P['ens_bio']['min_rate'],P['ens_bio']['max_rate']))
		ens_dir=nengo.Ensemble(n_neurons=1,dimensions=P['ens_bio']['dim'],
								neuron_type=nengo.Direct(),label='ens_dir')
		ens_bio2=nengo.Ensemble(n_neurons=P['ens_bio2']['n_neurons'],dimensions=P['ens_bio2']['dim'],
								neuron_type=BahlNeuron(P,'ens_bio2'),label='ens_bio2',										
								seed=P['ens_bio2']['seed'],radius=P['ens_bio2']['radius'],
								max_rates=nengo.dists.Uniform(P['ens_bio2']['min_rate'],P['ens_bio2']['max_rate']))
		ens_lif2=nengo.Ensemble(n_neurons=P['ens_bio2']['n_neurons'],dimensions=P['ens_bio2']['dim'],
								neuron_type=nengo.LIF(),label='ens_lif2',
								seed=P['ens_bio2']['seed'],radius=P['ens_bio2']['radius'],
								max_rates=nengo.dists.Uniform(P['ens_bio2']['min_rate'],P['ens_bio2']['max_rate']))
		ens_dir2=nengo.Ensemble(n_neurons=1,dimensions=P['ens_bio2']['dim'],
								neuron_type=nengo.Direct(),label='ens_dir2')

		'''SOLVERS'''
		solver_ens_bio=BioneuronSolver(P,ens_bio,method=P['decoder_train'])
		solver_ens_lif=nengo.solvers.LstsqL2()
		solver_ens_bio2=BioneuronSolver(P,ens_bio2,method=P['decoder_train'])
		solver_ens_lif2=nengo.solvers.LstsqL2()

		'''CONNECTIONS'''																
		nengo.Connection(stim,pre,synapse=None)
		nengo.Connection(pre,ens_bio,
							synapse=P['ens_bio']['tau'],
							transform=P['transform_pre_to_ens'])
		nengo.Connection(pre,ens_lif,
							synapse=P['ens_bio']['tau'],
							transform=P['transform_pre_to_ens'])
		nengo.Connection(stim,ens_dir,
							synapse=P['ens_bio']['tau'],
							transform=P['transform_pre_to_ens'])

		nengo.Connection(ens_bio,ens_bio,
							solver=solver_ens_bio,
							synapse=P['ens_bio']['tau'],
							transform=P['transform_ens_to_ens'])
		nengo.Connection(ens_lif,ens_lif,
							solver=solver_ens_lif,
							synapse=P['ens_bio']['tau'],
							transform=P['transform_ens_to_ens'])
		nengo.Connection(ens_dir,ens_dir,
							synapse=P['ens_bio']['tau'],
							transform=P['transform_ens_to_ens'])

		nengo.Connection(ens_bio,ens_bio2,
							synapse=P['ens_bio']['tau'],
							solver=solver_ens_bio,
							transform=P['transform_ens_to_ens2'])
		nengo.Connection(ens_lif,ens_lif2,
							synapse=P['ens_bio']['tau'],
							solver=solver_ens_lif,
							transform=P['transform_ens_to_ens2'])
		nengo.Connection(ens_dir,ens_dir2,
							synapse=P['ens_bio']['tau'],
							transform=P['transform_ens_to_ens2'])



		'''PROBES'''
		probe_stim=nengo.Probe(stim,synapse=None)
		probe_pre=nengo.Probe(pre,synapse=P['kernel']['tau'])
		probe_pre_spikes=nengo.Probe(pre.neurons,'spikes')
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

	with nengo.Simulator(model,
						pre_build_func=pre_build_func,
						post_build_func=post_build_func,
						dt=P['dt_nengo']) as sim:
		sim.run(P['test']['t_final'])

	if P['platform'] == 'workstation':
		sns.set(context='poster')
		os.chdir(P['directory'])
		figure1, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
		ax1.plot(sim.trange(),sim.data[probe_stim],label='stim')
		ax2.plot(sim.trange(),sim.data[probe_pre],label='pre_to_ens_1')
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
	
		figure2, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex=True)
		rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax1,use_eventplot=True)
		ax1.set(ylabel='bioneuron',yticks=([]),title='rmse (rates) = %.5f'%rmse_rates)
		rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax2,use_eventplot=True)
		ax2.set(ylabel='lif',yticks=([]))
		rasterplot(sim.trange(),sim.data[probe_bio_spikes2],ax=ax3,use_eventplot=True)
		ax3.set(ylabel='bioneuron2',yticks=([]),title='rmse (rates) = %.5f'%rmse_rates2)
		rasterplot(sim.trange(),sim.data[probe_lif_spikes2],ax=ax4,use_eventplot=True)
		ax4.set(ylabel='lif2',yticks=([]))
		figure2.savefig('bioneuron_vs_LIF_activity.png')
	
		try:
			os.makedirs('bioneuron_plots')
			os.chdir('bioneuron_plots')
		except:
			os.chdir('bioneuron_plots')
		figure4,ax1=plt.subplots(1,1)
		for nrn in range(len(ens_bio.neuron_type.father_op.neurons.neurons)):
			neuron=ens_bio.neuron_type.father_op.neurons.neurons[nrn]
			figure,ax=plt.subplots(1,1)
			ax.plot(P['dt_neuron']*np.arange(0,len(np.array(neuron.v_record))),np.array(neuron.v_record))
			ax.set(xlabel='time (ms)', ylabel='Voltage (mV)')
			figure.savefig('bioneuron_%s_voltages_test.png'%nrn)
			plt.close(figure)
			figure2,ax2=plt.subplots(1,1)
			bio_rates_plot=ax2.plot(sim.trange(),bio_rates[:,nrn],linestyle='-')
			ideal_rates_plot=ax2.plot(sim.trange(),lif_rates[:,nrn],linestyle='--',
				color=bio_rates_plot[0].get_color())
			ax2.plot(0,0,color='k',linestyle='-',label='bioneuron')
			ax2.plot(0,0,color='k',linestyle='--',label='LIF')
			rmse=np.sqrt(np.average((bio_rates[:,nrn]-lif_rates[:,nrn])**2))
			ax2.set(xlabel='time (s)',ylabel='firing rate (Hz)',title='rmse=%.5f'%rmse)
			figure2.savefig('bio_vs_ideal_rates_neuron_%s'%nrn)
			plt.close(figure2)
	
	elif P['platform'] == 'sharcnet':
		os.chdir(P['directory'])
		rmse_bio=np.sqrt(np.average((sim.data[probe_dir]-sim.data[probe_bio])**2))
		rmse_lif=np.sqrt(np.average((sim.data[probe_dir]-sim.data[probe_lif])**2))
		file=open('rmse_decode.txt','w')
		file.write('rmse_decode_bio')
		file.write(str(rmse_bio))
		file.write('rmse_decode_lif')
		file.write(str(rmse_lif))
		file.close()

		bio_rates=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_bio_spikes][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_bio_spikes].shape[1])]).T
		lif_rates=np.array([nengo.Lowpass(P['kernel']['tau']).filt(
						sim.data[probe_lif_spikes][:,nrn],dt=P['dt_nengo'])
						for nrn in range(sim.data[probe_lif_spikes].shape[1])]).T
		rmse_rates=np.sqrt(np.average((bio_rates-lif_rates)**2))
		file=open('rmse_test.txt','w')
		file.write('rmse_test_bio')
		file.write(str(rmse_rates))
		file.close()

if __name__=='__main__':
	main()

