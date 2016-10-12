def get_rates(P,bioneuron,LIFdata,addon,hyp_params):
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd
	import ipdb

	timesteps=P['timesteps']
	spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
	spike_train=np.zeros_like(timesteps)
	if spike_times.shape[0] >= 20000: ipdb.set_trace()
	for idx in spike_times/P['dt']/1000: spike_train[idx]=1.0
	rates=np.zeros_like(spike_train)

	if P['kernel']['type'] == 'exp':
		kernel = np.exp(-timesteps/P['kernel']['tau'])
		rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]
	elif P['kernel']['type'] == 'gauss':
		tkern = np.arange(-timesteps[-1]/4,timesteps[-1]/4,P['dt'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		rates = np.convolve(kernel, spike_train, mode='same')
	elif P['kernel']['type'] == 'alpha':  
		kernel = (timesteps / P['kernel']['tau']) * np.exp(-timesteps / P['kernel']['tau'])
		rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]
	elif P['kernel']['type'] == 'isi_smooth':
		f=isi_hold_function(timesteps,spike_times,midpoint=False)
		interp=f(spike_times)
		tkern = np.arange(-timesteps[-1]/4,timesteps[-1]/4,P['dt'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		rates = np.convolve(kernel, interp, mode='full')[:len(timesteps)]
	# kernel /= kernel.sum()



	sns.set(context='poster')
	figure, (ax1,ax2) = plt.subplots(2,1)
	# figure, (ax1,ax2,ax3) = plt.subplots(3,1)
	ax1.plot(np.array(bioneuron.t_record)/1000, np.array(bioneuron.v_record))
	ax1.set(xlabel='time', ylabel='voltage (mV)')
	ax2.plot(timesteps,LIFdata['signal_in'],label='input signal')
	for n in range(P['n_LIF']):
		ax2.plot(timesteps,np.array(LIFdata['spikes_in'])[:,n],label='input spikes [%s]'%n)
	# ax2.plot(timesteps,spike_train,label='output spikes')
	ax2.plot(timesteps,rates,label='output rate')
	ax2.set(xlabel='time (s)')
	# ax3.plot(timesteps,kernel,label='kernel')
	# ax3.set(xlabel='time (s)', ylabel='filter value')
	plt.legend()
	figure.savefig(hyp_params['directory']+addon+'_spikes.png')
	plt.close(figure)
	return rates

def make_tuning_curves(P,LIFdata,rates):
	import numpy as np
	X=np.arange(np.min(LIFdata['signal_in']),np.max(LIFdata['signal_in']),P['dx']) #eval points in X
	Hz=np.zeros_like(X) #firing rate for each eval point
	for xi in range(len(X)-1):
		ts=[] #find the time indices where the signal is between this and the next evalpoint
		for ti in range(len(P['timesteps'])):
			if X[xi] < LIFdata['signal_in'][ti] < X[xi+1]:
				ts.append(ti)
		if len(ts)>0:
			#average the firing rate at each of these time indices
			Hz[xi]=np.average([rates[ti] for ti in ts])
			#convert units to Hz by dividing by the time window
			Hz[xi]=Hz[xi]/len(ts)/P['dt']
	return X, Hz

def plot_loss(trials,hyp_params):
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1, 1)
	X=[t['tid'] for t in trials]
	Y=[t['result']['loss'] for t in trials]
	ax1.scatter(X,Y)
	ax1.set(xlabel='$t$',ylabel='loss')
	figure1.savefig(hyp_params['directory']+'hyperopt_result.png')


def isi_hold_function(t, spike_times, midpoint=False, interp='zero'):
	import numpy as np
	import scipy as sp
	"""
	Eric Hunsberger 2016 Tech Report
	Estimate firing rate using ISIs, with zero-order interpolation
	t : the times at which raw spike data (spikes) is defined
	spikes : the raw spike data
	midpoint : place interpolation points at midpoint of ISIs. Otherwise,
	    the points are placed at the beginning of ISIs
	"""
	isis = np.diff(spike_times)
	print isis
	if midpoint:
		rt = np.zeros(len(isis)+2)
		rt[0] = t[0]
		rt[1:-1] = 0.5*(spike_times[0:-1] + spike_times[1:])
		rt[-1] = t[-1]
		r = np.zeros_like(rt)
		r[1:-1] = 1. / isis
	else:
		rt = np.zeros(len(spike_times)+2)
		rt[0] = t[0]
		rt[1:-1] = spike_times
		rt[-1] = t[-1]
		r = np.zeros_like(rt)
		r[1:-2] = 1. / isis
	return sp.interpolate.interp1d(rt, r, kind=interp, copy=False)