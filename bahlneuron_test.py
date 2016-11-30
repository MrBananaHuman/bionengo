import nengo
import numpy as np
from BahlNeuron import BahlNeuron, CustomSolver
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb
import json

def signal(t,dim):
	return [np.sin((t+d**2*np.pi/(2.0/6))*2*np.pi/(1.0/6)) for d in range(dim)]

def main():
	P={
		'dt_nengo':0.001,
		'dt_neuron':0.0001,
		'filenames':None, #todo: bug when t_sim > t_sample and filenames=None
		# 'filenames':'/home/psipeter/bionengo/data/UJRQGR5ZS/filenames.txt', #with gain, bias
		# 'filenames':'/home/pduggins/bionengo/'+'data/U41GEJX4F/'+'filenames.txt', #with gain, bias
		'n_in':50,
		'n_bio':10,
		'n_syn':5,
		'dim':2,
		'ens_in_seed':333,
		'n_eval_points':3333,
		't_sim':1.0,

		'kernel_type':'gaussian',
		'tau_filter':0.01, #0.01, 0.2
		'min_ideal_rate':40,
		'max_ideal_rate':60,
		'signal': #for optimization and decoder calculation
			{'type':'equalpower','max_freq':15.0,'mean':0.0,'std':1.0},
		'kernel': #for smoothing spikes to calculate A matrix
			#{'type':'exp','tau':0.02},
			{'type':'gauss','sigma':0.01,},
		'synapse_tau':0.01,
		'synapse_type':'ExpSyn',

		'evals':10,
		't_train':1.0,
		'w_0':0.0005,
		'bias_min':-3.0,
		'bias_max':3.0,
		'n_seg': 5,
		'n_processes':10,
	}

	with nengo.Network() as model:
		stim=nengo.Node(output=lambda t: signal(t,P['dim']))
		ens_in=nengo.Ensemble(n_neurons=P['n_in'],dimensions=P['dim'],seed=P['ens_in_seed'])
		ens_bio=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=BahlNeuron(P),
								n_eval_points=P['n_eval_points'],label='ens_bio')
		test_lif=nengo.Ensemble(n_neurons=P['n_bio'],dimensions=P['dim'],
								neuron_type=nengo.LIF(),
								max_rates=nengo.dists.Uniform(P['min_ideal_rate'],
									P['max_ideal_rate']))
		ens_out=nengo.Ensemble(n_neurons=P['n_in'],dimensions=P['dim'])
		test_out=nengo.Ensemble(n_neurons=P['n_in'],dimensions=P['dim'])

		nengo.Connection(stim,ens_in,synapse=None)
		conn=nengo.Connection(ens_in,ens_bio,synapse=0.01)
		nengo.Connection(ens_in,test_lif,synapse=0.01)
		# solver=CustomSolver(P,model,conn)
		# nengo.Connection(ens_bio,ens_out,solver=solver,synapse=0.01)
		nengo.Connection(test_lif,test_out)

		probe_in=nengo.Probe(ens_in.neurons,'spikes')
		probe_voltage=nengo.Probe(ens_bio.neurons,'voltage')
		probe_spikes=nengo.Probe(ens_bio.neurons,'spikes')
		probe_bio=nengo.Probe(ens_bio,synapse=0.01)
		probe_test=nengo.Probe(test_lif,synapse=0.01)
		probe_lif_spikes=nengo.Probe(test_lif.neurons,'spikes')
		probe_out=nengo.Probe(ens_out,synapse=0.01)
		probe_test_out=nengo.Probe(test_out,synapse=0.01)

	with nengo.Simulator(model,dt=P['dt_nengo']) as sim:
		sim.run(P['t_sim'])


	sns.set(context='poster')
	solver=CustomSolver(P,model,conn)
	x_in=np.array(signal(sim.trange(),P['dim'])).T
	xhat_bio_out=np.dot(solver.A.T,solver.decoders)
	# xhat_bio_out=sim.data[probe_out]
	xhat_test_out=sim.data[probe_test_out]
	figure1, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,sharex=True)
	rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
	ax1.set(ylabel='input \nspikes')
	ax2.plot(sim.trange(),sim.data[probe_voltage])
	ax2.set(ylabel='bioneuron \nvoltage')
	# rasterplot(sim.trange(),sim.data[probe_spikes],ax=ax3,use_eventplot=True)
	# ax3.set(ylabel='bioneuron \nspikes',yticks=([]))
	rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax3,use_eventplot=True)
	ax3.set(ylabel='lif \nspikes',yticks=([]))
	ax4.plot(sim.trange(),x_in,label='$x(t)$') #color='k',ls='-',
	ax4.plot(sim.trange(),sim.data[probe_bio],label='bioneuron $\hat{x}(t)$')
	ax4.plot(sim.trange(),sim.data[probe_test],label='LIF $\hat{x}(t)$')
	ax4.set(ylabel='decoded \nens')
	ax5.plot(sim.trange(),x_in,label='$x(t)$') #color='k',ls='-',
	ax5.plot(sim.trange(),xhat_bio_out,label='bioneuron $\hat{x}(t)$ ')
	ax5.plot(sim.trange(),xhat_test_out,label='LIF $\hat{x}(t)$')
	ax5.set(xlabel='time (s)',ylabel='decoded \nens_out')
	plt.legend(loc='lower left')
	figure1.savefig('bioneuron_plots.png')

	figure2, ax2=plt.subplots(1,1)
	ax2.plot(x_in,x_in)
	ax2.plot(x_in,xhat_bio_out,label='bioneuron, RMSE=%s'
				%np.sqrt(np.average((x_in-xhat_bio_out)**2)))
	ax2.plot(x_in,xhat_test_out,label='LIF, RMSE=%s'
				%np.sqrt(np.average((x_in-xhat_test_out)**2)))
	ax2.set(xlabel='$x$',ylabel='$\hat{x}$')
	plt.legend(loc='lower right')
	figure2.savefig('rmse.png')

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