def get_rates(P,spikes):
	import numpy as np
	import ipdb
	import timeit

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
			if idx >= len(spike_train): break
			spike_train[idx]=1.0/P['dt']
	else:
		return np.zeros_like(timesteps), np.zeros_like(timesteps)

	# start2=timeit.default_timer()
	rates=np.zeros_like(spike_train)
	if P['kernel']['type'] == 'exp':
		tkern = np.arange(0,P['t_sample']/20.0,P['dt'])
		kernel = np.exp(-tkern/P['kernel']['tau'])
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]
	elif P['kernel']['type'] == 'gauss':
		tkern = np.arange(-P['t_sample']/20.0,P['t_sample']/20.0,P['dt'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='same')
	elif P['kernel']['type'] == 'alpha':  
		tkern = np.arange(0,P['t_sample']/20.0,P['dt'])
		kernel = (tkern / P['kernel']['tau']) * np.exp(-tkern / P['kernel']['tau'])
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]
	elif P['kernel']['type'] == 'isi_smooth':
		f=isi_hold_function(timesteps,spike_times,midpoint=False)
		interp=f(spike_times)
		tkern = np.arange(-P['t_sample']/20.0,P['t_sample']/20.0,P['dt'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		kernel /= kernel.sum()
		rates = np.convolve(kernel, interp, mode='full')[:len(timesteps)]
	# stop2=timeit.default_timer()
	# print 'part2 - %s sec' %(stop2-start2)
	return spike_train, rates


def make_tuning_curves(P,signal_in,biorates):
	import numpy as np
	import ipdb
	# X=np.arange(np.min(signal_in),np.max(signal_in),P['dx']) #eval points in X
	X=np.arange(-1.0,1.0,P['dx']) #eval points in X
	Hz=np.zeros_like(X) #firing rate for each eval point
	timesteps=np.arange(0,P['t_sample'],P['dt'])
	for xi in range(len(X)-1):
		ts_greater=np.where(X[xi] < signal_in)[0]
		ts_smaller=np.where(signal_in < X[xi+1])[0]
		ts=np.intersect1d(ts_greater,ts_smaller)
		if ts.shape[0]>0: Hz[xi]=np.average(biorates[ts])
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
	f_lif_rate = interp1d(lif_eval_points,lif_activities)
	x_min=np.maximum(lif_eval_points[0],bio_eval_points[0])
	x_max=np.minimum(lif_eval_points[-1],bio_eval_points[-1])
	X=np.arange(x_min,x_max,P['dx'])
	loss=np.sqrt(np.average((f_bio_rate(X)-f_lif_rate(X))**2))
	return X,f_bio_rate,f_lif_rate,loss


def plot_rates(P,loss,bioneuron,biospikes,biorates,raw_signal,lif_spikes,run_id):
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd
	import ipdb
	from nengo.utils.matplotlib import rasterplot

	timesteps=np.arange(0,P['t_sample'],P['dt'])
	sns.set(context='poster')
	figure, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
	ax1.plot(timesteps,raw_signal,label='input signal')
	ax1.set(ylabel='input signal')
	ax2.set(ylabel='LIF spikes')
	rasterplot(timesteps, lif_spikes,ax=ax2,use_eventplot=True)
	ax3.plot(np.array(bioneuron.t_record)/1000, np.array(bioneuron.v_record))
	ax3.set(ylabel='bioneuron mV')
	ax4.plot(timesteps,biorates,label='output rate')
	ax4.set(xlabel='time (s)',ylabel='bioneuron Hz')
	plt.legend()
	figure.savefig('loss=%0.3f_'%loss+run_id+'_spikes.png')
	plt.close(figure)


def plot_tuning_curve(P,X,f_bio_rate,f_lif_rate,loss,run_id):
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
	figure.savefig('loss=%0.3f_'%loss+run_id+'_tuning_curve.png')
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
	figure1.savefig(P['directory']+'bioneuron_%s_hyperopt_result.png'%P['bio_idx'])

def export_bioneuron(P,run_id,spike_times,loss):
	import json
	import numpy as np
	import ipdb
	bias=P['bias']
	weights=np.zeros((P['n_lif'],P['n_syn']))
	locations=np.zeros((P['n_lif'],P['n_syn']))
	for n in range(P['n_lif']):
		for i in range(P['n_syn']):
			weights[n][i]=P['weights']['%s_%s'%(n,i)]
			locations[n][i]=P['locations']['%s_%s'%(n,i)]
	to_export={
		'bio_idx':P['bio_idx'],
		'weights':weights.tolist(),
		'locations': locations.tolist(),
		'bias': bias,
		'spike_times': spike_times.tolist(),
		'loss': loss,
		}
	with open(run_id+'_bioneuron_%s.json'%P['bio_idx'], 'w') as data_file:
		json.dump(to_export, data_file)

# def make_dataframe(P,b,run_id,weights,locations,bias,spike_times,loss):
# 	import numpy as np
# 	import pandas as pd
# 	columns=('b','run_id','weight','location','bias','loss')
# 	df=pd.DataFrame(columns=columns,index=np.arange(0,P['n_lif']*P['n_syn']))
# 	for n in range(P['n_lif']):
# 		for i in range(P['n_syn']):
# 			df.loc[n*P['n_syn']+i]=[b,run_id,weights[n][i],locations[n][i],bias,spike_times,loss]
# 	return df

# def analyze_df(P,trials):
# 	import pandas as pd
# 	df=pd.concat([pd.DataFrame.from_csv(t['result']['run_id']+'_dataframe') \
# 		for t in trials],ignore_index=True)
# 	df.to_pickle(P['directory']+'dataframe.pkl')
# 	plot_weight_dist(P,df)
# 	return df

# def plot_weight_dist(P,df):
# 	import numpy as np
# 	import matplotlib.pyplot as plt
# 	import seaborn as sns
# 	import ipdb
# 	losses=np.sort(np.unique(df['loss']))
# 	cutoff=losses[np.ceil(P['loss_cutoff']*len(losses))-1] #top X% of losses
# 	weights=df.query("loss<=@cutoff")['weight']
# 	sns.set(context='poster')
# 	figure1, ax1 = plt.subplots(1, 1)
# 	sns.distplot(weights,kde=True,ax=ax1)
# 	ax1.set(title='location=%s (%s), w_0=%s, losses<=%s, N=%s'
# 							%(P['l_0'],P['synapse_dist'],P['w_0'],cutoff,len(weights)))
# 	figure1.savefig(P['directory']+'_weight_distribution.png')

def get_min_loss_filename(P,trials):
	import numpy as np
	import ipdb
	import json
	losses=[t['result']['loss'] for t in trials]
	ids=[t['result']['run_id'] for t in trials]
	idx=np.argmin(losses)
	best_run_id=str(ids[idx])
	filename=P['directory']+best_run_id+"_bioneuron_%s.json"%P['bio_idx']
	return filename

def plot_final_tuning_curves(P,filenames):
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import ipdb
	import json

	lifdata=np.load(P['directory']+'lifdata.npz')
	signal_in=lifdata['signal_in']
	spikes_in=lifdata['spikes_in']
	lif_eval_points=lifdata['lif_eval_points'].ravel()
	lif_activities=lifdata['lif_activities']

	biopop=[]
	for filename in filenames:
		with open(filename,'r') as data_file: 
			bioneuron=json.load(data_file)
		biopop.append(bioneuron)

	losses=[]
	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1, 1)
	ax1.set(xlabel='x',ylabel='firing rate (Hz)')
	for bio_idx in range(P['n_bio']):
		biospikes, biorates=get_rates(P,np.array(biopop[bio_idx]['spike_times']))
		bio_eval_points, bio_activities = make_tuning_curves(P,signal_in,biorates)
		X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(
				P,lif_eval_points,lif_activities[:,bio_idx],
				bio_eval_points,bio_activities)
		lifplot=ax1.plot(X,f_bio_rate(X),linestyle='-')
		bioplot=ax1.plot(X,f_lif_rate(X),linestyle='--',color=lifplot[0].get_color())
		losses.append(loss)
	ax1.set(ylim=(0,60),title='total loss = %s'%np.sum(losses))
	figure1.savefig('biopop_tuning_curves.png')
	plt.close(figure1)
	return np.sum(losses)