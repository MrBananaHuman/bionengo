import nengo
import numpy as np
from BahlNeuron import BahlNeuron#, CustomSolver
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb
import json

def signal(t,dim):
	return [np.sin((t+d**2*np.pi/(2.0/6))*2*np.pi/(1.0/6)) for d in range(dim)]
	# return t

# def rate_decoders_opt(filenames):
# 	f=open(filenames,'r')
# 	files=json.load(f)
# 	A_ideal=[]
# 	A_actual=[]
# 	gain_ideal=[]
# 	bias_ideal=[]
# 	x_sample=[]
# 	for bio_idx in range(len(files)):
# 		with open(files[bio_idx],'r') as data_file: 
# 			bioneuron_info=json.load(data_file)
# 		A_ideal.append(bioneuron_info['A_ideal'])
# 		A_actual.append(bioneuron_info['A_actual'])
# 		gain_ideal.append(bioneuron_info['gain_ideal'])
# 		bias_ideal.append(bioneuron_info['bias_ideal'])
# 		x_sample.append(bioneuron_info['x_sample'])
	
# 	solver=nengo.solvers.LstsqL2()
# 	decoders,info=solver(np.array(A_actual).T,np.array(x_sample)[0])
# 	# decoders,info=solver(np.array(A_ideal).T,np.array(x_sample)[0])
# 	return decoders

def main():
	dt_nengo=0.001
	dt_neuron=0.0001
	filenames=None #TODO: bug when t_sim > t_sample and filenames=None
	# filenames='/home/psipeter/bionengo/data/UJRQGR5ZS/filenames.txt' #with gain, bias
	# filenames='/home/pduggins/bionengo/'+'data/U41GEJX4F/'+'filenames.txt' #with gain, bias
	n_in=50
	n_bio=10
	n_syn=5
	dim=2
	evals=100
	t_sim=1.0
	kernel_type='gaussian'
	tau_filter=0.01 #0.01, 0.2

	with nengo.Network() as model:
		stim=nengo.Node(output=lambda t: signal(t,dim))
		ens_in=nengo.Ensemble(n_neurons=n_in,dimensions=dim,seed=333)
		ens_bio=nengo.Ensemble(n_neurons=n_bio,dimensions=dim,
								neuron_type=BahlNeuron(filenames),label='ens_bio')
		test_lif=nengo.Ensemble(n_neurons=n_bio,dimensions=dim,
								neuron_type=nengo.LIF(),max_rates=nengo.dists.Uniform(40,60))
		ens_out=nengo.Ensemble(n_neurons=n_in,dimensions=dim)
		test_out=nengo.Ensemble(n_neurons=n_in,dimensions=dim)

		nengo.Connection(stim,ens_in,synapse=None)
		nengo.Connection(ens_in,ens_bio,synapse=0.01)
		nengo.Connection(ens_in,test_lif,synapse=0.01)
		#need to pass filenames around properly
		# nengo.Connection(ens_bio.neurons,ens_out,
		# 						transform=np.ones((1,n_bio))*rate_decoders_opt(filenames),
		# 						synapse=0.01)
		# solver=CustomSolver(filenames)
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

	with nengo.Simulator(model,dt=dt_nengo) as sim:
		sim.run(t_sim)


	sns.set(context='poster')
	x_in=np.array(signal(sim.trange(),dim)).T
	xhat_bio_out=sim.data[probe_out]
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

	figure3, ax3 = plt.subplots(1, 1)
	for bio_idx in range(n_bio):
		lifplot=ax3.plot(solver.x_sample[bio_idx,:-2],solver.A_ideal[bio_idx,:-2],linestyle='--')
		bioplot=ax3.plot(solver.x_sample[bio_idx,:-2],solver.A_actual[bio_idx,:-2],linestyle='-',
							color=lifplot[0].get_color())
	ax3.plot(0,0,color='k',linestyle='-',label='bioneuron')
	ax3.plot(0,0,color='k',linestyle='--',label='LIF')
	ax3.set(xlabel='x',ylabel='firing rate (Hz)',ylim=(0,60))
	plt.legend(loc='upper center')
	figure3.savefig('response_curve_comparison.png')

if __name__=='__main__':
	main()