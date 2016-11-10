import nengo
import numpy as np
from BioneuronNode import BioneuronNode
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot

def get_spike_train(spike_times):
	biospikes=[]
	for bio_idx in range(len(spike_times)):
		spike_train=np.zeros_like(sim.trange())
		for t_idx in np.array(spike_times[bio_idx])/dt_nengo:
			if t_idx >= len(spike_train): break
			spike_train[t_idx]=1.0/dt_nengo
		biospikes.append(spike_train)
	biospikes=np.array(biospikes).T
	return biospikes

def calculate_decoders_standard(bionode,noise=0.1,function=lambda x: x):
	A=np.matrix(bionode.A_ideal)
	A_T=np.matrix(bionode.A_ideal.T)
	# A=np.matrix(bionode.A_actual)
	# A_T=np.matrix(bionode.A_actual.T)
	f_X=np.matrix(function(bionode.x_sample)).T
	S=len(bionode.x_sample)
	upsilon=A_T*f_X/S
	gamma=A_T*A/S + np.identity(bionode.n_bio)*(noise*np.max(A))**2
	d=np.linalg.inv(gamma)*upsilon
	#test
	# x=np.array(f_X)
	# xhat=np.array(np.dot(A,d)).ravel()
	# figure, ax1=plt.subplots(1,1)
	# ax1.plot(x,x)
	# ax1.plot(x,xhat)
	# ax1.set(xlabel='$x$',ylabel='$\hat{x}$',title='RMSE=%s'
	# 	%np.sqrt(np.average((x-xhat)**2)))
	# plt.show()
	return d

def rate_estimate(bionode,d,kern):
	spike_train=np.array(bionode.spike_train)
	d=np.array(d)
	timesteps=spike_train.shape[0]
	n_neurons=d.shape[0]
	dimension=d.shape[1]
	xhat=np.zeros((timesteps,dimension))
	for i in range(n_neurons):
		filtered_spikes=np.convolve(spike_train[:,i],kern,mode='full')[:timesteps].reshape(timesteps,1)
		value=filtered_spikes*d[i]
		xhat+=value
	return xhat

dt_nengo=0.0001
dt_neuron=0.0001
filenames=None
filenames='/home/pduggins/bionengo/'+'data/M1AMCVG2Q/'+'filenames.txt'
n_in=50
n_bio=5
n_syn=5
evals=1000
t_sim=0.1
with nengo.Network() as model:
	stim=nengo.Node(output=lambda t: np.sin(t*2*np.pi/t_sim))
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

d=calculate_decoders_standard(bionode,noise=0.1,function=lambda x: x)
xhat=rate_estimate(bionode,d,kernel)

sns.set(context='poster')
figure, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
voltages=np.array([bioneuron.nengo_voltages for bioneuron in bionode.biopop]).T
spike_times=np.array([bioneuron.nengo_spike_times for bioneuron in bionode.biopop]).T
biospikes=get_spike_train(spike_times)
rasterplot(sim.trange(), sim.data[probe_in],ax=ax1,use_eventplot=True)
ax1.set(ylabel='lif spikes')
ax2.plot(sim.trange(),voltages)
ax2.set(ylabel='bioneuron voltage \n(dt_nengo)')
rasterplot(sim.trange(),biospikes,ax=ax3,use_eventplot=True)
ax3.set(xlabel='time (s)', ylabel='bioneuron spikes \n(dt_nengo)')
ax4.plot(sim.trange(),np.sin(sim.trange()*2*np.pi/t_sim))
ax4.plot(sim.trange(),xhat)
ax4.set(xlabel='time (s)', ylabel='$\hat{x}$ bionode')
plt.show()