import nengo
import numpy as np
from BioneuronNode import BioneuronNode
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot

def decoders_from_tuning_curve(bionode,noise=0.1,function=lambda x: x):
	# A=np.matrix(bionode.A_ideal)
	# A_T=np.matrix(bionode.A_ideal.T)
	A=np.matrix(bionode.A_actual)
	A_T=np.matrix(bionode.A_actual.T)
	f_X=np.matrix(function(bionode.x_sample)).T
	# S=len(bionode.x_sample)
	# upsilon=A_T*f_X/S
	# gamma=A_T*A/S + np.identity(bionode.n_bio)*(noise*np.max(A))**2
	# d=np.linalg.inv(gamma)*upsilon
	# np.savez('A_and_evals_from_tuning_curve.npz',A=A,evals=bionode.x_sample)
	solver=nengo.solvers.LstsqL2()
	d,info=solver(np.array(A),np.array(f_X))
	return d, np.array(f_X), np.array(A)


def decoders_from_simulation(bionode, kernel_type, tau_filter, sim, function=lambda x: x):
	spike_train=np.array(bionode.spike_train)
	# np.savez('spikes_from_sin_input.npz',spike_train=spike_train)
	if kernel_type == 'lowpass': 
		lowpass_filter=nengo.synapses.Lowpass(tau=tau_filter)
		A_sample=lowpass_filter.filt(spike_train)
		f_X=function(lowpass_filter.filt(signal(sim.trange())))
	elif kernel_type == 'gaussian': 
		tkern = np.arange(-t_sim/20.0,t_sim/20.0,dt_nengo)
		kernel = np.exp(-tkern**2/(2*tau_filter**2))
		kernel /= kernel.sum()
		A_sample=np.array([np.convolve(kernel,spike_train[:,i],mode='same')
							for i in range(len(bionode.biopop))]).T
		f_X=function(np.convolve(kernel, signal(sim.trange()), mode='same'))
	solver=nengo.solvers.LstsqL2()
	d,info=solver(A_sample,f_X)
	return d, f_X, A_sample

def rate_estimate(bionode,d,kernel_type, tau_filter):
	# ipdb.set_trace()
	spike_train=np.array(bionode.spike_train)
	d=np.array(d)
	timesteps=spike_train.shape[0]
	n_neurons=d.shape[0]
	if kernel_type == 'lowpass': 
		lowpass_filter=nengo.synapses.Lowpass(tau=tau_filter)
		A=lowpass_filter.filt(spike_train)
	elif kernel_type == 'gaussian': 
		tkern = np.arange(-t_sim/20.0,t_sim/20.0,dt_nengo)
		kernel = np.exp(-tkern**2/(2*tau_filter**2))
		kernel /= kernel.sum()
		A=np.array([np.convolve(kernel,spike_train[:,i],mode='same')
						for i in range(n_neurons)]).T
	xhat=np.array(np.dot(A,d)).ravel()
	return xhat, A

def lif_comparison(test_lif,sim,stt,kernel_type, tau_filter):
	#Test smoothing/decoding procedure on a normal LIF population
	x_test, A_test=nengo.utils.ensemble.tuning_curves(test_lif,sim)
	solver_test=nengo.solvers.LstsqL2()
	d_test,info_test=solver_test(np.array(A_test),np.array(x_test))
	spike_train_test=stt
	if kernel_type == 'lowpass': 
		lowpass_filter=nengo.synapses.Lowpass(tau=tau_filter)
		A_test=lowpass_filter.filt(spike_train_test)
	elif kernel_type == 'gaussian': 
		tkern = np.arange(-t_sim/20.0,t_sim/20.0,dt_nengo)
		kernel = np.exp(-tkern**2/(2*tau_filter**2))
		kernel /= kernel.sum()
		A_test=np.array([np.convolve(kernel,spike_train_test[:,i],mode='same') 
						for i in range(stt.shape[1])]).T
	xhat_test=np.array(np.dot(A_test,d_test))
	return spike_train_test, A_test, xhat_test

def signal(t):
	return np.sin(t*2*np.pi/(t_sim/2))


def main():
	global dt_nengo
	dt_nengo=0.0001
	dt_neuron=0.0001
	filenames=None
	# filenames='/home/pduggins/bionengo/'+'data/0NP48SK8A/'+'filenames.txt'
	# filenames='/home/pduggins/bionengo/'+'data/4NX39QJ32/'+'filenames.txt'
	n_in=50
	n_bio=2
	n_syn=5
	evals=1000
	global t_sim
	t_sim=1.0
	kernel_type='gaussian'
	tau_filter=0.01 #0.01, 0.2
	with nengo.Network() as model:
		stim=nengo.Node(output=lambda t: signal(t))
		ens_in=nengo.Ensemble(n_neurons=n_in,dimensions=1,seed=333)
		bionode=BioneuronNode(n_in=n_in,n_bio=n_bio,n_syn=n_syn,
								dt_nengo=dt_nengo,dt_neuron=dt_neuron)
		# test_lif=nengo.Ensemble(n_neurons=n_bio,dimensions=1)
		test_lif=nengo.Ensemble(n_neurons=2,dimensions=1,
								encoders=[[1],[-1]],
								max_rates=[80,80],
								intercepts=[-0.25,-0.25])
		nengo.Connection(stim,ens_in,synapse=None)
		bionode.connect_to(ens_in.seed,evals=evals,filenames=filenames)
		nengo.Connection(ens_in.neurons,bionode,synapse=None)
		nengo.Connection(ens_in,test_lif,synapse=None)

		probe_in=nengo.Probe(ens_in.neurons,'spikes')
		probe_bio=nengo.Probe(bionode)
		probe_test=nengo.Probe(test_lif.neurons,'spikes')
	with nengo.Simulator(model,dt=dt_nengo) as sim:
		sim.run(t_sim)


	d,x_sample,A_sample=decoders_from_tuning_curve(bionode,function=lambda x: x)
	# d,x_sample,A_sample=decoders_from_simulation(
	# 					bionode, kernel_type,tau_filter, sim, function=lambda x: x)
	xhat,A=rate_estimate(
						bionode,d,kernel_type,tau_filter) #kernel=kernel
	spike_train_test, A_test, xhat_test = lif_comparison(
						test_lif,sim,sim.data[probe_test],kernel_type,tau_filter)
	xhat_sample=np.array(np.dot(A_sample,d)).ravel()
	spike_train=np.array(bionode.spike_train)


	sns.set(context='poster')
	figure1, (ax1,ax2,ax5) = plt.subplots(3,1,sharex=True)
	# figure1, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,sharex=True)
	# figure2, (ax20,ax21,ax22,ax23) = plt.subplots(4,1,sharex=True)
	voltages=np.array([bioneuron.nengo_voltages for bioneuron in bionode.biopop]).T
	rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
	ax1.set(ylabel='input spikes',title='Bioneuron')
	ax2.plot(sim.trange(),voltages)
	ax2.set(ylabel='voltage')
	# rasterplot(sim.trange(),spike_train,ax=ax3,use_eventplot=True)
	# ax3.set(ylabel='spikes')
	# for i in range(A.shape[1]):
	# 	ax4.plot(sim.trange(),A[:,i],label='bio')
	# ax4.set(ylabel='firing rate (Hz)')
	ax5.plot(sim.trange(),signal(sim.trange()),color='k',ls='-',label='$x(t)$')
	ax5.plot(sim.trange(),xhat,color='k',ls='--',label='$\hat{x}(t)$')
	ax5.set(xlabel='time (s)', ylabel='$\hat{x}$')
	plt.legend()

	figure2, (ax20,ax21,ax23) = plt.subplots(3,1,sharex=True)
	rasterplot(sim.trange(), sim.data[probe_in],ax=ax20,use_eventplot=True)
	ax20.set(ylabel='input spikes',title='LIF')
	rasterplot(sim.trange(),spike_train_test,ax=ax21,use_eventplot=True)
	ax21.set(ylabel='output spikes')
	# for i in range(A.shape[1]):
	# 	ax22.plot(sim.trange(),A_test[:,i],label='lif_test')
	# ax22.set(ylabel='firing rate (Hz)')
	ax23.plot(sim.trange(),signal(sim.trange()),color='k',ls='-',label='$x(t)$')
	ax23.plot(sim.trange(),xhat_test,color='k',ls='--',label='$\hat{x}(t)$')
	ax23.set(xlabel='time (s)', ylabel='$\hat{x}$')
	plt.legend()

	figure3, ax1=plt.subplots(1,1)
	ax1.plot(x_sample,x_sample)
	ax1.plot(x_sample,xhat_sample)
	ax1.set(xlabel='$x$',ylabel='$\hat{x}$',title='RMSE=%s'
		%np.sqrt(np.average((x_sample-xhat_sample)**2)))

	figure3.savefig('rmse.png')
	figure1.savefig('bioneuron_plots.png')
	figure2.savefig('lif_plots.png')
	# plt.show()

if __name__=='__main__':
	main()