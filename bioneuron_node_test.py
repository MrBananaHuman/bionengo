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
	np.savez('A_and_evals_from_tuning_curve.npz',A=A_actual,evals=bionode.x_sample)
	solver=nengo.solvers.LstsqL2()
	d,info=solver(np.array(A),np.array(f_X))
	return d, np.array(f_X), np.array(A)

def decoders_from_spikes(n_in,n_bio,n_syn,dt_nengo,dt_neuron,evals,filenames,
						kernel,t_sample):
	def decoder_signal(t):
		return np.sin(t*2*np.pi/t_sample)	
	with nengo.Network() as decoder_model:
		stim2=nengo.Node(output=lambda t: decoder_signal(t))
		ens_in2=nengo.Ensemble(n_neurons=n_in,dimensions=1)
		decoder_bionode=BioneuronNode(n_in=n_in,n_bio=n_bio,n_syn=n_syn,
							dt_nengo=dt_nengo,dt_neuron=dt_neuron,
							evals=evals,filenames=filenames)
		nengo.Connection(stim2,ens_in2)
		nengo.Connection(ens_in2.neurons,decoder_bionode)
	with nengo.Simulator(decoder_model,dt=dt_nengo) as decoder_sim:
		decoder_sim.run(t_sample)
	A_rates=[]
	spike_train=np.array(decoder_bionode.spike_train)
	np.savez('spikes_from_sin_input.npz',spike_train=spike_train)
	for i in range(spike_train.shape[1]):
		rates = np.convolve(kernel, spike_train[:,i], mode='same')
		A_rates.append(rates)
	A_rates=np.array(A_rates).T
	target_filtered=np.convolve(kernel, decoder_signal(decoder_sim.trange()), mode='same')
	solver=nengo.solvers.LstsqL2()
	# decoders,info=solver(A_rates,decoder_signal(decoder_sim.trange()))
	decoders,info=solver(A_rates,target_filtered)
	return decoders, target_filtered, A_rates

def rate_estimate(bionode,d,kernel):
	# ipdb.set_trace()
	spike_train=np.array(bionode.spike_train)
	d=np.array(d)
	timesteps=spike_train.shape[0]
	n_neurons=d.shape[0]
	if len(d.shape) > 1:
		dimension=d.shape[1]
	else:
		dimension=1
	A=np.array([np.convolve(kernel,spike_train[:,i],mode='same') for i in range(n_neurons)]).T
	xhat=np.array(np.dot(A,d)).ravel()
	return xhat, A

def signal(t):
	return np.sin(t*2*np.pi/t_sim)

dt_nengo=0.0001
dt_neuron=0.0001
filenames=None
filenames='/home/pduggins/bionengo/'+'data/M1AMCVG2Q/'+'filenames.txt'
n_in=50
n_bio=5
n_syn=5
evals=1000
t_sim=0.5
with nengo.Network() as model:
	stim=nengo.Node(output=lambda t: signal(t))
	ens_in=nengo.Ensemble(n_neurons=n_in,dimensions=1)
	bionode=BioneuronNode(n_in=n_in,n_bio=n_bio,n_syn=n_syn,
							dt_nengo=dt_nengo,dt_neuron=dt_neuron,
							evals=evals,filenames=filenames)
	# ens_out=nengo.Ensemble(n_neurons=n_in,dimensions=1)
	nengo.Connection(stim,ens_in)
	nengo.Connection(ens_in.neurons,bionode)
	# nengo.Connection(bionode,ens_out)
	probe_in=nengo.Probe(ens_in.neurons,'spikes')
	probe_bio=nengo.Probe(bionode)
	# probe_out=nengo.Probe(ens_out)
with nengo.Simulator(model,dt=dt_nengo) as sim:
	sim.run(t_sim)

#set post-synaptic current temporal filter
sigma=0.01          
tkern = np.arange(-t_sim/20.0,t_sim/20.0,dt_nengo)
kernel = np.exp(-tkern**2/(2*sigma**2))
kernel /= kernel.sum()
# d,x_sample,A_sample=decoders_from_tuning_curve(bionode,noise=0.1,function=lambda x: x)
d,x_sample,A_sample=decoders_from_spikes(n_in,n_bio,n_syn,dt_nengo,dt_neuron,
						evals,filenames,kernel,0.5)
xhat,A=rate_estimate(bionode,d,kernel)

sns.set(context='poster')
xhat_sample=np.array(np.dot(A_sample,d)).ravel()
figureA, ax1=plt.subplots(1,1)
ax1.plot(x_sample,x_sample)
ax1.plot(x_sample,xhat_sample)
ax1.set(xlabel='$x$',ylabel='$\hat{x}$',title='RMSE=%s'
	%np.sqrt(np.average((x_sample-xhat_sample)**2)))
# plt.show()

figure, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,sharex=True)
voltages=np.array([bioneuron.nengo_voltages for bioneuron in bionode.biopop]).T
spike_train=np.array(bionode.spike_train)
rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
ax1.set(ylabel='lif spikes')
ax2.plot(sim.trange(),voltages)
ax2.set(ylabel='bio voltage \n(dt_nengo)')
rasterplot(sim.trange(),spike_train,ax=ax3,use_eventplot=True)
ax3.set(ylabel='bio spikes \n(dt_nengo)')
for i in range(A.shape[1]):
	ax4.plot(sim.trange(),A[:,i])
ax4.set(ylabel='bio activities')
ax5.plot(sim.trange(),signal(sim.trange()),label='$x(t)$')
ax5.plot(sim.trange(),xhat,label='$\hat{x}(t)$')
ax5.set(xlabel='time (s)', ylabel='$\hat{x}$ bionode')
plt.legend()
plt.show()