'''
Initialization 
###################################################################################################
'''

def make_addon(N):
	import string
	import random
	addon=str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(N)))
	return addon

def ch_dir():
	#change directory for data and plot outputs
	import os
	import sys
	root=os.getcwd()
	addon=make_addon(9)
	datadir=''
	if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
		datadir=root+'/data/'+addon+'/' #linux or mac
	elif sys.platform == "win32":
		datadir=root+'\\data\\'+addon+'\\' #windows
	os.makedirs(datadir)
	os.chdir(datadir) 
	return datadir

def make_signal(P):
	""" Returns: array indexed by t when called from a nengo Node"""
	import signals
	sP=P['signal']
	dt=P['dt']
	t_final=P['t_sample']+dt #why is this extra step necessary?
	raw_signal=None
	if sP['type']=='constant':
		raw_signal=signals.constant(dt,t_final,sP['value'])
	if sP['type']=='white':
		raw_signal=signals.white(dt,t_final)
	elif sP['type']=='white_binary':
		raw_signal=signals.white_binary(dt,t_final,sP['mean'],sP['std'])
	elif sP['type']=='switch':
		raw_signal=signals.switch(dt,t_final,sP['max_freq'],)
	elif sP['type']=='equalpower':
		raw_signal=signals.equalpower(
			dt,t_final,sP['max_freq'],sP['mean'],sP['std'])
	elif sP['type']=='poisson_binary':
		raw_signal=signals.poisson_binary(
			dt,t_final,sP['mean_freq'],sP['max_freq'],sP['low'],sP['high'])
	elif sP['type']=='poisson':
		raw_signal=signals.poisson(
			dt,t_final,sP['mean_freq'],sP['max_freq'])
	elif sP['type']=='pink_noise':
		raw_signal=signals.pink_noise(
			dt,t_final,sP['mean'],sP['std'])
	assert raw_signal is not None, "signal type not specified"
	return raw_signal

def make_spikes_in(P,raw_signal):
	import nengo
	import numpy as np
	import pandas as pd
	# import json
	spikes_in=[]
	lifdata={}
	while np.sum(spikes_in)==0: #rerun nengo spike generator until it returns something
		with nengo.Network() as model:
			signal = nengo.Node(
					output=lambda t: raw_signal[int(t/P['dt'])])
			ideal = nengo.Ensemble(P['n_bio'],
					dimensions=1, #ideal tuning curve has limited rate
					max_rates=nengo.dists.Uniform(P['min_in_rate'],P['max_lif_rate']))
			ens_in = nengo.Ensemble(P['n_in'],
					dimensions=1)		
			nengo.Connection(signal,ens_in)
			probe_signal = nengo.Probe(signal)
			probe_in = nengo.Probe(ens_in.neurons,'spikes')
		with nengo.Simulator(model,dt=P['dt']) as sim:
			sim.run(P['t_sample'])
			eval_points, activities = nengo.utils.ensemble.tuning_curves(ideal,sim)
		signal_in=sim.data[probe_signal]
		spikes_in=sim.data[probe_in]
	np.savez(P['directory']+'lifdata.npz',
			signal_in=signal_in.ravel(),spikes_in=spikes_in,
			lif_eval_points=eval_points,lif_activities=activities,
			gains=ideal.gain, biases=ideal.bias)

def weight_rescale(location):
	#interpolation
	import numpy as np
	from scipy.interpolate import interp1d
	#load voltage attenuation data for bahl.hoc
	voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
	f_voltage_att = interp1d(voltage_attenuation['distances'],voltage_attenuation['voltages'])
	scaled_weight=1.0/f_voltage_att(location)
	return scaled_weight

def add_search_space(P,bio_idx):
	#adds a hyperopt-distributed weight, location, bias for each synapse
	import numpy as np
	import hyperopt
	P['bio_idx']=bio_idx
	P['weights']={}
	P['locations']={}
	P['bias']=hyperopt.hp.uniform('b',P['bias_min'],P['bias_max'])
	for n in range(P['n_in']):
		for i in range(P['n_syn']): 
			P['locations']['%s_%s'%(n,i)] =\
				np.round(np.random.uniform(0.0,1.0),decimals=2)
			k_weight=weight_rescale(P['locations']['%s_%s'%(n,i)])
			P['weights']['%s_%s'%(n,i)]=\
				hyperopt.hp.uniform('w_%s_%s'%(n,i),-k_weight*P['w_0'],k_weight*P['w_0'])
	return P	

'''
ANALYSIS 
###################################################################################################
'''

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
	return spike_train, rates


def make_tuning_curves(P,signal_in,biorates):
	import numpy as np
	import ipdb
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
	#shape of activities and Hz may be mismatched,
	#so interpolate and slice activities for comparison
	from scipy.interpolate import interp1d
	f_bio_rate = interp1d(bio_eval_points,bio_activities)
	f_lif_rate = interp1d(lif_eval_points,lif_activities)
	x_min=np.maximum(lif_eval_points[0],bio_eval_points[0])
	x_max=np.minimum(lif_eval_points[-1],bio_eval_points[-1])
	X=np.arange(x_min,x_max,P['dx'])
	loss=np.sqrt(np.average((f_bio_rate(X)-f_lif_rate(X))**2))
	return X,f_bio_rate,f_lif_rate,loss

def export_bioneuron(P,run_id,spike_times,X,f_bio_rate,f_lif_rate,loss):
	import json
	import numpy as np
	import ipdb
	bias=P['bias']
	weights=np.zeros((P['n_in'],P['n_syn']))
	locations=np.zeros((P['n_in'],P['n_syn']))
	for n in range(P['n_in']):
		for i in range(P['n_syn']):
			weights[n][i]=P['weights']['%s_%s'%(n,i)]
			locations[n][i]=P['locations']['%s_%s'%(n,i)]
	to_export={
		'bio_idx':P['bio_idx'],
		'weights':weights.tolist(),
		'locations': locations.tolist(),
		'bias': bias,
		'spike_times': spike_times.tolist(),
		'x_sample': X.tolist(),
		'A_ideal': f_lif_rate(X).tolist(),
		'A_actual': f_bio_rate(X).tolist(),
		'loss': loss,
		}
	with open(run_id+'_bioneuron_%s.json'%P['bio_idx'], 'w') as data_file:
		json.dump(to_export, data_file)

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

def get_best_biopop(P,filenames):
	import json
	biopop=[]
	for filename in filenames:
		with open(filename,'r') as data_file: 
			bioneuron=json.load(data_file)
		biopop.append(bioneuron)
	return biopop

def plot_final_tuning_curves(P,biopop):
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
	plt.close('all')


'''
BIONEURON METHODS 
###################################################################################################
'''


def make_bioneuron(P,weights,locations,bias):
	import numpy as np
	from neurons import Bahl
	bioneuron=Bahl()
	#make connections and populate with synapses
	for n in range(P['n_in']):
		bioneuron.add_bias(bias)
		bioneuron.add_connection(n)
		for i in range(P['n_syn']):
			syn_type=P['synapse_type']
			section=bioneuron.cell.apical(locations[n][i])
			weight=weights[n][i]
			tau=P['synapse_tau']
			bioneuron.add_synapse(n,syn_type,section,weight,tau)
	#initialize recording attributes
	bioneuron.start_recording()
	return bioneuron

def connect_bioneuron(P,spikes_in,bioneuron):
	import numpy as np
	import neuron
	import ipdb
	for n in range(P['n_in']):
		#create spike time vectors and an artificial spiking cell that delivers them
		vstim=neuron.h.VecStim()
		bioneuron.vecstim[n]['vstim'].append(vstim)
		spike_times_ms=list(1000*P['dt']*np.nonzero(spikes_in[:,n])[0]) #timely
		vtimes=neuron.h.Vector(spike_times_ms)
		bioneuron.vecstim[n]['vtimes'].append(vtimes)
		bioneuron.vecstim[n]['vstim'][-1].play(bioneuron.vecstim[n]['vtimes'][-1])
		#connect the VecStim to each synapse
		for syn in bioneuron.synapses[n]:
			netcon=neuron.h.NetCon(bioneuron.vecstim[n]['vstim'][-1],syn.syn)
			netcon.weight[0]=abs(syn.weight)
			bioneuron.netcons[n].append(netcon)

def run_bioneuron(P):
	import neuron
	neuron.h.dt = P['dt']*1000
	neuron.init()
	neuron.run(P['t_sample']*1000)


'''
MAIN 
###################################################################################################
'''


def run_hyperopt(P):
	import hyperopt
	import numpy as np
	trials=hyperopt.Trials()
	best=hyperopt.fmin(simulate,
		space=P,
		algo=hyperopt.tpe.suggest,
		max_evals=P['evals'],
		trials=trials)
	filename=get_min_loss_filename(P,trials)
	return filename

def simulate(P):
	import numpy as np
	import neuron
	import hyperopt
	import json
	import os
	import timeit
	import gc

	start=timeit.default_timer()
	run_id=make_addon(6)
	os.chdir(P['directory'])
	lifdata=np.load(P['directory']+'lifdata.npz')
	signal_in=lifdata['signal_in']
	spikes_in=lifdata['spikes_in']
	lif_eval_points=lifdata['lif_eval_points'].ravel()
	lif_activities=lifdata['lif_activities'][:,P['bio_idx']]
	bias=P['bias']
	weights=np.zeros((P['n_in'],P['n_syn']))
	locations=np.zeros((P['n_in'],P['n_syn']))
	for n in range(P['n_in']):
		for i in range(P['n_syn']):
			weights[n][i]=P['weights']['%s_%s'%(n,i)]
			locations[n][i]=P['locations']['%s_%s'%(n,i)]
	bioneuron = make_bioneuron(P,weights,locations,bias)
	connect_bioneuron(P,spikes_in,bioneuron)
	
	run_bioneuron(P)
	spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
	biospikes, biorates=get_rates(P,spike_times)
	bio_eval_points, bio_activities = make_tuning_curves(P,signal_in,biorates)	
	X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(
			P,lif_eval_points,lif_activities,bio_eval_points,bio_activities)
	export_bioneuron(P,run_id,spike_times,X,f_bio_rate,f_lif_rate,loss)
	del bioneuron
	gc.collect()
	stop=timeit.default_timer()
	print 'Simulate Runtime - %s sec' %(stop-start)
	return {'loss': loss, 'run_id':run_id, 'status': hyperopt.STATUS_OK}

def optimize_bioneuron(n_in,n_bio,n_syn,evals=1000,
					dt_neuron=0.0001,dt_nengo=0.001,
					tau=0.01,synapse_type='ExpSyn'):
	import json
	import copy
	from pathos.multiprocessing import ProcessingPool as Pool

	datadir=ch_dir()
	P={
		'directory':datadir,
		'n_in':n_in,
		'n_syn':n_syn,
		'n_bio':n_bio,
		'dt':dt_neuron,
		'evals':evals,
		'synapse_tau':tau,
		'synapse_type':synapse_type,
		't_sample':1.0,
		'min_in_rate':40,
		'max_lif_rate':60,
		'w_0':0.0005,
		'bias_min':-3.0,
		'bias_max':3.0,
		'n_seg': 5,
		'dx':0.05,
		'n_processes':n_bio,
		'signal':
			{'type':'equalpower','max_freq':10.0,'mean':0.0,'std':1.0},
			#{'type':'constant','value':1.0},
			#{'type':'pink_noise','mean':0.0,'std':1.0},
			#{'type':'poisson','mean_freq':5.0,'max_freq':10.0},
			#{'type':'switch','max_freq':10.0,},
			#{'type':'white_binary','mean':0.0,'std':1.0},
			#{'type':'poisson_binary','mean_freq':5.0,'max_freq':10.0,'low':-1.0,'high':1.0},
			#{'type':'white'},
		'kernel': #for smoothing spikes to calculate rate for tuning curve
			#{'type':'exp','tau':0.02},
			{'type':'gauss','sigma':0.01,},
			#{'type':'alpha','tau':0.1},
	}

	raw_signal=make_signal(P)
	make_spikes_in(P,raw_signal)
	P_list=[]
	pool = Pool(nodes=P['n_processes'])
	for bio_idx in range(P['n_bio']):
		P_idx=add_search_space(P,bio_idx)
		P_list.append(copy.copy(P_idx))
	filenames=pool.map(run_hyperopt, P_list)
	with open('filenames.txt','wb') as outfile:
		json.dump(filenames,outfile)
	biopop_dict=get_best_biopop(P,filenames)
	plot_final_tuning_curves(P,biopop_dict)
	return P['directory']+'filenames.txt'