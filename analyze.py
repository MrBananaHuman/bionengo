def get_rates(P,spikes):
	import numpy as np
	import ipdb

	timesteps=np.arange(0,P['t_sample'],P['dt'])
	if spikes.shape[0]==len(timesteps): #if we're given a spike train
		if spikes.ndim == 2: #sum over all neurons' spikes
			spike_train=np.sum(spikes,axis=1)
		else:
			spike_train=spikes
	elif len(spikes) > 0: #if we're given spike times and there's at least one spike
		spike_train=np.zeros_like(timesteps)
		spike_times=spikes.ravel()
		for idx in spike_times/P['dt']/1000:
			spike_train[idx]=1.0/P['dt']
	else:
		return np.zeros_like(timesteps), np.zeros_like(timesteps)
	
	rates=np.zeros_like(spike_train)
	if P['kernel']['type'] == 'exp':
		kernel = np.exp(-timesteps/P['kernel']['tau'])
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]
	elif P['kernel']['type'] == 'gauss':
		tkern = np.arange(-timesteps[-1]/4,timesteps[-1]/4,P['dt'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='same')
	elif P['kernel']['type'] == 'alpha':  
		kernel = (timesteps / P['kernel']['tau']) * np.exp(-timesteps / P['kernel']['tau'])
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]
	elif P['kernel']['type'] == 'isi_smooth':
		f=isi_hold_function(timesteps,spike_times,midpoint=False)
		interp=f(spike_times)
		tkern = np.arange(-timesteps[-1]/4,timesteps[-1]/4,P['dt'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		kernel /= kernel.sum()
		rates = np.convolve(kernel, interp, mode='full')[:len(timesteps)]
	return spike_train, rates


def make_tuning_curves(P,lifdata,biorates):
	import numpy as np
	import ipdb

	X=np.arange(np.min(lifdata['signal_in']),np.max(lifdata['signal_in']),P['dx']) #eval points in X
	Hz=np.zeros_like(X) #firing rate for each eval point
	for xi in range(len(X)-1):
		ts=[] #find the time indices where the signal is between this and the next evalpoint
		for ti in range(len(np.arange(0,P['t_sample'],P['dt']))):
			if X[xi] < lifdata['signal_in'][ti] < X[xi+1]:
				ts.append(ti)
		if len(ts)>0:
			#average the firing rate at each of these time indices
			Hz[xi]=np.average([biorates[ti] for ti in ts])
			#convert units to Hz by dividing by the time window
			# Hz[xi]=Hz[xi]/len(ts)
	return X, Hz


def tuning_curve_loss(P,lif_eval_points,lif_activities,bio_eval_points,bio_activities):
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd
	import json
	import ipdb

	#shape of activities and Hz is mismatched, so interpolate and slice activities for comparison
	from scipy.interpolate import interp1d
	f_bio_rate = interp1d(bio_eval_points,bio_activities)
	f_lif_rate = interp1d(np.array(lif_eval_points),np.array(lif_activities))
	x_min=np.maximum(lif_eval_points[0],bio_eval_points[0])
	x_max=np.minimum(lif_eval_points[-1],bio_eval_points[-1])
	X=np.arange(x_min,x_max,P['dx'])
	loss=np.sqrt(np.average((f_bio_rate(X)-f_lif_rate(X))**2))
	return X,f_bio_rate,f_lif_rate,loss


def plot_rates(P,bioneuron,biospikes,biorates,raw_signal,lif_spikes,run_id):
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd
	import ipdb

	timesteps=np.arange(0,P['t_sample'],P['dt'])
	sns.set(context='poster')
	figure, (ax1,ax2,ax3) = plt.subplots(3,1)
	ax1.plot(np.array(bioneuron.t_record)/1000, np.array(bioneuron.v_record))
	ax1.set(xlabel='time', ylabel='bioneuron voltage (mV)')
	ax2.plot(timesteps,raw_signal,label='input signal')
	for n in range(P['n_lif']):
		ax2.plot(timesteps,np.array(lif_spikes)[:,n]*P['dt'],label='input spikes [%s]'%n)
	ax2.plot(timesteps,biospikes*P['dt'],label='output spikes')
	ax2.set(xlabel='time (s)')
	ax3.plot(timesteps,biorates,label='output rate')
	ax3.set(xlabel='time (s)',ylabel='bioneuron Hz')
	plt.legend()
	figure.savefig(run_id+'_spikes.png')
	plt.close(figure)


def plot_tuning_curve(X,f_bio_rate,f_lif_rate,loss,run_id):
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd
	import json
	import ipdb

	sns.set(context='poster')
	figure, ax1 = plt.subplots(1,1)
	ax1.plot(X,f_bio_rate(X),label='bioneuron firing rate (Hz)')
	ax1.plot(X,f_lif_rate(X),label='lif firing rate (Hz)')
	ax1.set(xlabel='x',ylabel='firing rate (Hz)',title='loss=%0.3f' %loss)
	plt.legend()
	figure.savefig(run_id+'_tuning_curve.png')
	plt.close(figure)


def plot_loss(P,trials):
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1, 1)
	X=[t['tid'] for t in trials]
	Y=[t['result']['loss'] for t in trials]
	ax1.scatter(X,Y)
	ax1.set(xlabel='$t$',ylabel='loss')
	figure1.savefig(P['directory']+'hyperopt_result.png')


def export_params(P,run_id):
	import pandas as pd
	import json
	my_params=pd.DataFrame([P])
	my_params.reset_index().to_json(run_id+'_params.json',orient='records')

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