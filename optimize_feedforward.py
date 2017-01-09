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
	import numpy as np
	sP=P['signal']
	dt=P['dt_nengo']
	t_final=P['t_train']+dt #why is this extra step necessary?
	if sP['type']=='prime_sinusoids':
		raw_signal=signals.prime_sinusoids(dt,t_final,P['dim'])
		return raw_signal
	raw_signal=[]
	for d in range(P['dim']):
		if sP['type']=='constant':
			raw_signal.append(signals.constant(dt,t_final,sP['value']))
		elif sP['type']=='white':
			raw_signal.append(signals.white(dt,t_final))
		elif sP['type']=='white_binary':
			raw_signal.append(signals.white_binary(dt,t_final,sP['mean'],sP['std']))
		elif sP['type']=='switch':
			raw_signal.append(signals.switch(dt,t_final,sP['max_freq'],))
		elif sP['type']=='equalpower':
			raw_signal.append(signals.equalpower(
				dt,t_final,sP['max_freq'],sP['mean'],sP['std']))
		elif sP['type']=='poisson_binary':
			raw_signal.append(signals.poisson_binary(
				dt,t_final,sP['mean_freq'],sP['max_freq'],sP['low'],sP['high']))
		elif sP['type']=='poisson':
			raw_signal.append(signals.poisson(
				dt,t_final,sP['mean_freq'],sP['max_freq']))
		elif sP['type']=='pink_noise':
			raw_signal.append(signals.pink_noise(
				dt,t_final,sP['mean'],sP['std']))
	assert len(raw_signal) > 0, "signal type not specified"
	return np.array(raw_signal)

def generate_ideal_spikes(P,raw_signal):
	import nengo
	import numpy as np
	import pandas as pd
	import ipdb
	with nengo.Network() as opt_model:
		signal = nengo.Node(lambda t: raw_signal[:,int(t/P['dt_nengo'])]) #all dim, index at t
		pre = nengo.Ensemble(n_neurons=P['ens_pre_neurons'],
								dimensions=P['ens_pre_dim'],
								max_rates=nengo.dists.Uniform(P['ens_pre_min_rate'],
																P['ens_pre_max_rate']),
								seed=P['ens_pre_seed'],radius=P['ens_pre_radius'],)
		ideal = nengo.Ensemble(n_neurons=P['ens_ideal_neurons'],
								dimensions=P['ens_ideal_dim'],
								max_rates=nengo.dists.Uniform(P['ens_ideal_min_rate'],
																P['ens_ideal_max_rate']),
								seed=P['ens_ideal_seed'],radius=P['ens_ideal_radius'],)
		nengo.Connection(signal,pre,synapse=None)
		nengo.Connection(pre,ideal,synapse=P['tau'])
		probe_signal = nengo.Probe(signal)
		probe_pre = nengo.Probe(pre.neurons,'spikes')
		probe_ideal = nengo.Probe(ideal.neurons,'spikes')
	with nengo.Simulator(opt_model,dt=P['dt_nengo']) as opt_sim:
		opt_sim.run(P['t_train'])
		# eval_points, activities = nengo.utils.ensemble.tuning_curves(ideal,opt_sim)
	gains=opt_sim.data[ideal].gain
	biases=opt_sim.data[ideal].bias
	encoders=opt_sim.data[ideal].encoders
	signal_in=opt_sim.data[probe_signal]
	spikes_in=opt_sim.data[probe_pre]
	spikes_ideal=opt_sim.data[probe_ideal]
	np.savez(P['inputs']+'lifdata.npz',
			signal_in=signal_in,spikes_in=spikes_in,spikes_ideal=spikes_ideal,
			gains=gains, biases=biases, encoders=encoders)
			# ideal_eval_points=eval_points,ideal_activities=activities)

def weight_rescale(location):
	#interpolation
	import numpy as np
	from scipy.interpolate import interp1d
	#load voltage attenuation data for bahl.hoc
	# voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
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
	for n in range(P['ens_pre_neurons']):
		for i in range(P['n_syn']): 
			P['locations']['%s_%s'%(n,i)] =\
				np.round(np.random.uniform(0.0,1.0),decimals=2)
			#scale max weight to statistics of conn.pre ensemble and synapse location:
			#farther along dendrite means higher max weights allowed
			#fewer neurons and lower max_rates means higher max weights allowed
			#todo: make explict scaling functions for these relationships
			k_distance=weight_rescale(P['locations']['%s_%s'%(n,i)])
			k_neurons=50.0/P['ens_pre_neurons']
			k_max_rates=300.0/np.average([P['ens_pre_min_rate'],P['ens_pre_max_rate']])
			k=k_distance*k_neurons*k_max_rates
			P['weights']['%s_%s'%(n,i)]=hyperopt.hp.uniform('w_%s_%s'%(n,i),-k*P['w_0'],k*P['w_0'])
	return P	

'''
ANALYSIS 
###################################################################################################
'''

def get_rates(P,spikes):
	import numpy as np
	import ipdb
	import nengo
	timesteps=np.arange(0,P['t_train'],P['dt_nengo'])
	if spikes.shape[0]==len(timesteps): #if we're given a spike train
		if spikes.ndim == 2: #sum over all neurons' spikes
			spike_train=np.sum(spikes,axis=1)
		else:
			spike_train=spikes
	elif len(spikes) > 0: #if we're given spike times and there's at least one spike
		spike_train=np.zeros_like(timesteps)
		spike_times=spikes.ravel()
		for idx in spike_times/P['dt_nengo']/1000:
			if idx >= len(spike_train): break
			spike_train[int(idx)]=1.0/P['dt_nengo']
	else:
		return np.zeros_like(timesteps), np.zeros_like(timesteps)
	rates=np.zeros_like(spike_train)
	if P['kernel']['type'] == 'lowpass':
		lpf=nengo.Lowpass(P['kernel']['tau'])
		rates=lpf.filt(spike_train,dt=P['dt_nengo'])
	elif P['kernel']['type'] == 'exp':
		tkern = np.arange(0,P['t_train']/20.0,P['dt_nengo'])
		kernel = np.exp(-tkern/P['kernel']['tau'])
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]
	elif P['kernel']['type'] == 'gauss':
		tkern = np.arange(-P['t_train']/20.0,P['t_train']/20.0,P['dt_nengo'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='same')
	elif P['kernel']['type'] == 'alpha':  
		tkern = np.arange(0,P['t_train']/20.0,P['dt_nengo'])
		kernel = (tkern / P['kernel']['tau']) * np.exp(-tkern / P['kernel']['tau'])
		kernel /= kernel.sum()
		rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]
	elif P['kernel']['type'] == 'isi_smooth':
		f=isi_hold_function(timesteps,spike_times,midpoint=False)
		interp=f(spike_times)
		tkern = np.arange(-P['t_train']/20.0,P['t_train']/20.0,P['dt_nengo'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		kernel /= kernel.sum()
		rates = np.convolve(kernel, interp, mode='full')[:len(timesteps)]
	return spike_train, rates

def rate_loss(bio_rates,ideal_rates):
	import numpy as np
	loss=np.sqrt(np.average((bio_rates-ideal_rates)**2))
	return loss

def compare_bio_ideal_rates(P,filenames):
	import numpy as np
	import ipdb
	import json
	import matplotlib.pyplot as plt
	import seaborn as sns
	from nengo.utils.matplotlib import rasterplot
	biopop=[]
	for filename in filenames:
		with open(filename,'r') as data_file: 
			bioneuron=json.load(data_file)
		biopop.append(bioneuron)
	sns.set(context='poster')
	figure1, (ax0,ax1,ax2,ax3) = plt.subplots(4, 1,sharex=True)

	timesteps=np.arange(0,P['t_train'],P['dt_nengo'])
	spikes_in=np.array(biopop[0]['spikes_in'])
	bio_spikes=np.array([np.array(biopop[b]['bio_spikes']) for b in range(len(biopop))]).T
	ideal_spikes=np.array([np.array(biopop[b]['ideal_spikes']) for b in range(len(biopop))]).T
	bio_rates=np.array([np.array(biopop[b]['bio_rates']) for b in range(len(biopop))])
	ideal_rates=([np.array(biopop[b]['ideal_rates']) for b in range(len(biopop))])
	loss=np.sqrt(np.average((bio_rates-ideal_rates)**2))

	rasterplot(timesteps,spikes_in,ax=ax0,use_eventplot=True)
	rasterplot(timesteps,ideal_spikes,ax=ax1,use_eventplot=True)
	rasterplot(timesteps,bio_spikes,ax=ax2,use_eventplot=True)
	ax0.set(ylabel='input spikes',title='loss=%s'%loss)
	ax1.set(ylabel='ideal spikes')
	ax2.set(ylabel='bio spikes')
	for b in range(len(biopop)):
		bio_rates=ax3.plot(timesteps,np.array(biopop[b]['bio_rates']),linestyle='-')
		ideal_rates=ax3.plot(timesteps,np.array(biopop[b]['ideal_rates']),linestyle='--',
			color=bio_rates[0].get_color())
	ax3.plot(0,0,color='k',linestyle='-',label='bioneuron')
	ax3.plot(0,0,color='k',linestyle='--',label='LIF')
	ax3.set(xlabel='time (s)',ylabel='firing rate (Hz)',xlim=(0,1.0))
	plt.legend(loc='center right', prop={'size':6}, bbox_to_anchor=(1.1,0.8))
	figure1.savefig('a(t)_bio_vs_ideal.png')

# def make_tuning_curves(P,signal_in,encoders_ideal,biorates):
# 	import numpy as np
# 	import ipdb
# 	# X=np.arange(-1.0,1.0,P['dx']) #eval points in X
# 	#generate 1d evaluation points by dotting N dim signal with N dim ideal encoders
# 	min_x_dot_e=np.min(np.dot(signal_in,encoders_ideal.T))
# 	max_x_dot_e=np.max(np.dot(signal_in,encoders_ideal.T))
# 	X=np.arange(min_x_dot_e,max_x_dot_e,P['dx'])
# 	Hz=np.zeros_like(X) #firing rate for each eval point
# 	timesteps=np.arange(0,P['t_train'],P['dt_nengo'])
# 	for xi in range(len(X)-1):
# 		ts_greater=np.where(X[xi] < signal_in)[0]
# 		ts_smaller=np.where(signal_in < X[xi+1])[0]
# 		ts=np.intersect1d(ts_greater,ts_smaller)
# 		if ts.shape[0]>0: Hz[xi]=np.average(biorates[ts])
# 	return X, Hz

# def tuning_curve_loss(P,lif_eval_points,lif_activities,bio_eval_points,bio_activities):
# 	import numpy as np
# 	import ipdb
# 	#shape of activities and Hz may be mismatched,
# 	#so interpolate and slice activities for comparison
# 	from scipy.interpolate import interp1d
# 	f_bio_rate = interp1d(bio_eval_points,bio_activities)
# 	f_lif_rate = interp1d(lif_eval_points,lif_activities)
# 	x_min=np.maximum(np.min(bio_eval_points),np.min(lif_eval_points))
# 	x_max=np.minimum(np.max(bio_eval_points),np.max(lif_eval_points))
# 	#todo: generalize -1.0 to radius
# 	assert x_min < -1.0, "x_min=%s don't extend to radius, interpolation fail" %x_min
# 	assert x_max > 1.0, "x_max=%s don't extend to radius, interpolation fail" %xmax
# 	X=np.arange(-1.0,1.0,P['dx'])
# 	# X=np.arange(x_min,x_max,P['dx'])
# 	loss=np.sqrt(np.average((f_bio_rate(X)-f_lif_rate(X))**2))
# 	return X,f_bio_rate,f_lif_rate,loss

# def plot_final_tuning_curves(P,biopop):
# 	import numpy as np
# 	import matplotlib.pyplot as plt
# 	import seaborn as sns
# 	import ipdb
# 	import json
# 	lifdata=np.load(P['directory']+'lifdata.npz')
# 	signal_in=lifdata['signal_in']
# 	spikes_in=lifdata['spikes_in']
# 	lif_eval_points=lifdata['lif_eval_points'].ravel()
# 	lif_activities=lifdata['lif_activities']
# 	losses=[]
# 	sns.set(context='poster')
# 	figure1, ax1 = plt.subplots(1, 1)
# 	ax1.set(xlabel='x',ylabel='firing rate (Hz)')
# 	for bio_idx in range(P['n_bio']):
# 		biospikes, biorates=get_rates(P,np.array(biopop[bio_idx]['spike_times']))
# 		bio_eval_points, bio_activities = make_tuning_curves(P,signal_in,biorates)
# 		X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(
# 				P,lif_eval_points,lif_activities[:,bio_idx],
# 				bio_eval_points,bio_activities)
# 		lifplot=ax1.plot(X,f_bio_rate(X),linestyle='-')
# 		bioplot=ax1.plot(X,f_lif_rate(X),linestyle='--',color=lifplot[0].get_color())
# 		losses.append(loss)
# 	ax1.set(ylim=(0,60))
# 	figure1.savefig('biopop_tuning_curves.png')
# 	plt.close('all')


'''
BIONEURON METHODS 
###################################################################################################
'''


def make_bioneuron(P,weights,locations,bias):
	import numpy as np
	from neurons import Bahl
	bioneuron=Bahl()
	#make connections and populate with synapses
	for n in range(P['ens_pre_neurons']):
		bioneuron.add_bias(bias)
		bioneuron.add_connection(n)
		for i in range(P['n_syn']):
			syn_type=P['synapse_type']
			section=bioneuron.cell.apical(locations[n][i])
			weight=weights[n][i]
			tau=P['tau']
			bioneuron.add_synapse(n,syn_type,section,weight,tau)
	#initialize recording attributes
	bioneuron.start_recording()
	return bioneuron

def connect_bioneuron(P,spikes_in,bioneuron):
	import numpy as np
	import neuron
	import ipdb
	for n in range(P['ens_pre_neurons']):
		#create spike time vectors and an artificial spiking cell that delivers them
		vstim=neuron.h.VecStim()
		bioneuron.vecstim[n]['vstim'].append(vstim)
		spike_times_ms=list(1000*P['dt_nengo']*np.nonzero(spikes_in[:,n])[0]) #timely
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
	neuron.h.dt = P['dt_neuron']*1000
	neuron.init()
	neuron.run(P['t_train']*1000)

def export_bioneuron(P,run_id,bias,weights,locations,signal_in,spikes_in,
					ideal_spikes,bio_spikes,ideal_rates,bio_rates,loss):
	import json
	to_export={
		'bio_idx':P['bio_idx'],
		'weights':weights.tolist(),
		'locations': locations.tolist(),
		'bias': bias,
		# 'gain_ideal':gain_ideal,
		# 'bias_ideal':bias_ideal,
		# 'encoders_ideal':encoders_ideal.tolist(),
		'signal_in': signal_in.tolist(),
		'spikes_in':spikes_in.tolist(),
		'ideal_spikes': ideal_spikes.tolist(),
		'bio_spikes': bio_spikes.tolist(),
		'ideal_rates': ideal_rates.tolist(),
		'bio_rates': bio_rates.tolist(),
		'loss': loss,
		}
	with open(P['inputs']+run_id+
				'_bioneuron_%s.json'%P['bio_idx'], 'w') as data_file:
		json.dump(to_export, data_file)

'''
MAIN 
###################################################################################################
'''


def run_hyperopt(P):
	import hyperopt
	import numpy as np
	import json
	import ipdb
	trials=hyperopt.Trials()
	for t in range(P['evals']):
		best=hyperopt.fmin(simulate,
			space=P,
			algo=hyperopt.tpe.suggest,
			max_evals=(t+1),
			trials=trials)
		print 'Neuron %s hyperopt %s%%' %(P['bio_idx'],100.0*(t+1)/P['evals'])
	losses=[t['result']['loss'] for t in trials]
	ids=[t['result']['run_id'] for t in trials]
	idx=np.argmin(losses)
	best_run_id=str(ids[idx])
	filename=P['inputs']+best_run_id+"_bioneuron_%s.json"%P['bio_idx']
	return filename

def simulate(P):
	import numpy as np
	import neuron
	import hyperopt
	import json
	import os
	import timeit
	import gc
	import ipdb

	start=timeit.default_timer()
	run_id=make_addon(6)
	lifdata=np.load(P['inputs']+'lifdata.npz')
	signal_in=lifdata['signal_in']
	spikes_ideal=lifdata['spikes_ideal'][:,P['bio_idx']]
	if P['spikes_train'] == 'bio' and P['ens_pre_type']=='BahlNeuron()':
		biodata=np.load(P['directory']+P['ens_pre_label']+'/pre/biodata.npz') #todo: hardcoded path
		spikes_in=biodata['bio_spikes'].T
	else:
		spikes_in=lifdata['spikes_in']
	bias=P['bias']
	weights=np.zeros((P['ens_pre_neurons'],P['n_syn']))
	locations=np.zeros((P['ens_pre_neurons'],P['n_syn']))
	for n in range(P['ens_pre_neurons']):
		for i in range(P['n_syn']):
			weights[n][i]=P['weights']['%s_%s'%(n,i)]
			locations[n][i]=P['locations']['%s_%s'%(n,i)]

	bioneuron = make_bioneuron(P,weights,locations,bias)
	connect_bioneuron(P,spikes_in,bioneuron)
	run_bioneuron(P)
	
	bio_spikes, bio_rates=get_rates(P,np.round(np.array(bioneuron.spikes),decimals=3))
	ideal_spikes, ideal_rates=get_rates(P,spikes_ideal)
	loss = rate_loss(bio_rates,ideal_rates)
	export_bioneuron(P,run_id,bias,weights,locations,signal_in,spikes_in,
					ideal_spikes,bio_spikes,ideal_rates,bio_rates,loss)

	del bioneuron
	gc.collect()
	stop=timeit.default_timer()
	# print 'Simulate Runtime - %s sec' %(stop-start)
	return {'loss': loss, 'run_id':run_id, 'status': hyperopt.STATUS_OK}

def optimize_feedforward(P):
	import json
	import copy
	import ipdb
	import os
	from pathos.multiprocessing import ProcessingPool as Pool
	
	P=copy.copy(P)
	P['inputs']=P['directory']+P['ens_label']+'/'+P['ens_pre_label']+'/'
	if os.path.exists(P['inputs']):
		return P['inputs']
	raw_signal=make_signal(P)
	os.makedirs(P['inputs'])
	os.chdir(P['inputs'])
	generate_ideal_spikes(P,raw_signal) #feedforward connection
	P_list=[]
	pool = Pool(nodes=P['n_processes'])
	for bio_idx in range(P['n_bio']):
		P_idx=add_search_space(P,bio_idx)
		# f=run_hyperopt(P_idx)
		P_list.append(copy.copy(P_idx))
	filenames=pool.map(run_hyperopt, P_list)
	with open('filenames.txt','wb') as outfile:
		json.dump(filenames,outfile)
	compare_bio_ideal_rates(P,filenames)
	os.chdir(P['directory'])
	return P['inputs']