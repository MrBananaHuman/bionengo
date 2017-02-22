import numpy as np
import nengo
import neuron
import hyperopt
import timeit
import json
import copy
import ipdb
import os
import sys
import gc
import string
import random
import signals
import matplotlib.pyplot as plt
import seaborn
from synapses import ExpSyn
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
from pathos.multiprocessing import ProcessingPool as Pool
import pickle

def make_addon(N):
	addon=str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(N)))
	return addon

def ch_dir():
	#change directory for data and plot outputs
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
	#todo:cleanup
	""" Returns: array indexed by t when called from a nengo Node"""
	signal_type=P['type']
	dt=P['dt']
	t_final=P['t_final']+dt #why?
	dim=P['dim']
	if signal_type =='equalpower':
		mean=P['mean']
		std=P['std']
		max_freq=P['max_freq']
		seed=P['seed']
		if seed == None: seed=np.random.randint(99999999)
	if signal_type == 'constant':
		mean=P['mean']
	if signal_type=='sinusoid':
		omega=P['omega']
	raw_signal=[]
	for d in range(dim):
		if signal_type=='sinusoid':
			raw_signal.append(signals.sinusoid(dt,t_final,omega))
		if signal_type=='constant':
			raw_signal.append(signals.constant(dt,t_final,mean))
		elif signal_type=='white':
			raw_signal.append(signals.white(dt,t_final))
		elif signal_type=='white_binary':
			raw_signal.append(signals.white_binary(dt,t_final,mean,std))
		elif signal_type=='switch':
			raw_signal.append(signals.switch(dt,t_final,max_freq,))
		elif signal_type=='equalpower':
			raw_signal.append(signals.equalpower(dt,t_final,max_freq,mean,std,seed=seed))
		elif signal_type=='poisson_binary':
			raw_signal.append(signals.poisson_binary(dt,t_final,mean_freq,max_freq,low,high))
		elif signal_type=='poisson':
			raw_signal.append(signals.poisson(dt,t_final,mean_freq,max_freq))
		elif signal_type=='pink_noise':
			raw_signal.append(signals.pink_noise(dt,t_final,mean,std))
	assert len(raw_signal) > 0, "signal type not specified"
	#todo - scale to transform or radius
	return np.array(raw_signal)

def make_pre_and_ideal_spikes(P):
	import signals as sigz
	atrb=P['ens_post']['atrb']
	signals={}
	stims={}
	pres={}
	synapses={}
	transforms={}
	connections_pres={}
	connections_ideal={}
	p_stims={}
	p_pres={}		
	p_pres_spikes={}
	if P['optimize']['type']=='sinusoid':
		n_inputs=len(P['ens_post']['inpts'])
		primes=sigz.primeno(n_inputs)
		i=0
	with nengo.Network() as opt_model:
		ideal=nengo.Ensemble(
				label=atrb['label'],
				n_neurons=atrb['neurons'],
				dimensions=atrb['dim'],
				max_rates=nengo.dists.Uniform(atrb['min_rate'],atrb['max_rate']),
				radius=atrb['radius'],
				seed=atrb['seed'])
		p_ideal_spikes=nengo.Probe(ideal.neurons,'spikes')
		for key in P['ens_post']['inpts'].iterkeys():
			pre_atrb=P['ens_post']['inpts'][key]
			P_signal=copy.copy(P['optimize'])
			if P_signal['type']=='sinusoid':
				P_signal['omega']=primes[i]
				i+=1
			signals[key]=make_signal(P_signal)
			stims[key]=nengo.Node(lambda t, key=key: signals[key][:,np.floor(t/P['dt_nengo'])])
			pres[key]=nengo.Ensemble(
					label=pre_atrb['pre_label'],
					n_neurons=pre_atrb['pre_neurons'],
					dimensions=pre_atrb['pre_dim'],
					max_rates=nengo.dists.Uniform(pre_atrb['pre_min_rate'],pre_atrb['pre_max_rate']),
					radius=pre_atrb['pre_radius'],
					seed=pre_atrb['pre_seed'])
			transforms[key]=pre_atrb['transform']
			synapses[key]=pre_atrb['synapse']
			connections_pres[key]=nengo.Connection(
					stims[key],pres[key],
					synapse=None,transform=1.0)
			connections_ideal[key]=nengo.Connection(
					pres[key],ideal,
					synapse=synapses[key],transform=transforms[key])
			p_stims[key]=nengo.Probe(stims[key],synapse=None)
			p_pres[key]=nengo.Probe(pres[key],synapse=synapses[key])
			p_pres_spikes[key]=nengo.Probe(pres[key].neurons,'spikes')
	with nengo.Simulator(opt_model,dt=P['dt_nengo']) as opt_test:
		opt_test.run(P_signal['t_final'])
	all_signals_in=[]
	lpf=nengo.Lowpass(P['kernel']['tau'])
	for key in P['ens_post']['inpts'].iterkeys():
		# signal_in=transforms[key]*opt_test.data[p_pres[key]]
		signal_in=lpf.filt(
			# lpf.filt(
			transforms[key]*opt_test.data[p_stims[key]],
				# dt=P['dt_nengo']),
			dt=P['dt_nengo'])
		spikes_in=opt_test.data[p_pres_spikes[key]]
		all_signals_in.append(signal_in)
		np.savez('signals_spikes_from_%s_to_%s.npz'%(key,P['ens_post']['atrb']['label']),
						signal_in=signal_in,spikes_in=spikes_in)
	spikes_ideal=opt_test.data[p_ideal_spikes]
	np.savez('ideal_spikes_%s.npz'%P['ens_post']['atrb']['label'],spikes_ideal=spikes_ideal)
	np.savez('target_signal.npz',target_signal=np.sum(all_signals_in,axis=0))














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

def make_hyperopt_search_space_onebyone(P_in,my_P,bionrn,rng):
	#adds a hyperopt-distributed weight, location, bias for each synapse for each bioneuron,
	#where each neuron is a seperate choice in hyperopt search space
	P=copy.copy(P_in)
	hyperparams={}
	hyperparams['bionrn']=bionrn
	hyperparams['bias']=hyperopt.hp.uniform('b_%s'%bionrn,P['bias_min'],P['bias_max'])
	for inpt in P['ens_post']['inpts'].iterkeys():
		hyperparams[inpt]={}
		ens_pre_info=P['ens_post']['inpts'][inpt]
		for pre in range(ens_pre_info['pre_neurons']):
			hyperparams[inpt][pre]={}
			for syn in range(my_P['n_syn']):
				hyperparams[inpt][pre][syn]={}
				if P['optimize_locations']==True:
					hyperparams[inpt][pre][syn]['l']=hyperopt.hp.uniform('l_%s_%s_%s_%s'%(bionrn,inpt,pre,syn),0.0,1.0)
					k_distance=2.0
				else:
					hyperparams[inpt][pre][syn]['l']=np.round(rng.uniform(0.0,1.0),decimals=2)
					k_distance=weight_rescale(hyperparams[inpt][pre][syn]['l'])
				k_neurons=50.0/ens_pre_info['pre_neurons']
				k_max_rates=300.0/np.average([ens_pre_info['pre_min_rate'],ens_pre_info['pre_max_rate']])
				k=k_distance*k_neurons*k_max_rates
				hyperparams[inpt][pre][syn]['w']=hyperopt.hp.uniform('w_%s_%s_%s_%s'%(bionrn,inpt,pre,syn),-k*P['w_0'],k*P['w_0'])
	P['hyperopt']=hyperparams
	P['evals']=my_P['evals']
	P['tau']=my_P['tau']
	P['n_syn']=my_P['n_syn']
	return P

def load_signals_spikes(P):
	signals_spikes_pres={}
	for key in P['ens_post']['inpts'].iterkeys():
		signals_spikes_pres[key]=np.load('signals_spikes_from_%s_to_%s.npz'%(key,P['ens_post']['atrb']['label']))
	spikes_ideal=np.load('ideal_spikes_%s.npz'%P['ens_post']['atrb']['label'])['spikes_ideal']
	return signals_spikes_pres,spikes_ideal









class Bahl():
	def __init__(self,P,bias):
		# neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
		neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
		self.cell = neuron.h.Bahl()
		self.bias = bias
		self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
		self.bias_current.delay = 0
		self.bias_current.dur = 1e9  # TODO; limits simulation time
		self.bias_current.amp = self.bias
		self.synapses={}
		self.netcons={}
	def make_synapses(self,P,my_weights,my_locations):
		for inpt in P['ens_post']['inpts'].iterkeys():
			ens_pre_info=P['ens_post']['inpts'][inpt]
			self.synapses[inpt]=np.empty((ens_pre_info['pre_neurons'],P['n_syn']),dtype=object)
			# self.netcons[inpt]=np.empty((ens_pre_info['pre_neurons'],P['n_syn']),dtype=object)
			for pre in range(ens_pre_info['pre_neurons']):
				for syn in range(P['n_syn']):
					section=self.cell.apical(my_locations[inpt][pre][syn])
					weight=my_weights[inpt][pre][syn]
					synapse=ExpSyn(section,weight,P['tau'])
					self.synapses[inpt][pre][syn]=synapse	
	def start_recording(self):
		self.v_record = neuron.h.Vector()
		self.v_record.record(self.cell.soma(0.5)._ref_v)
		self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
		self.t_record = neuron.h.Vector()
		self.t_record.record(neuron.h._ref_t)
		self.spikes = neuron.h.Vector()
		self.ap_counter.record(neuron.h.ref(self.spikes))
	def event_step(self,t_neuron,inpt,pre):
		for syn in self.synapses[inpt][pre]: #for each synapse in this connection
			syn.spike_in.event(t_neuron) #add a spike at time (ms)





def load_hyperopt_weights_onebyone(P):
	weights={}
	locations={}
	bias=P['hyperopt']['bias']
	for inpt in P['ens_post']['inpts'].iterkeys():
		ens_pre_info=P['ens_post']['inpts'][inpt]
		weights[inpt]=np.zeros((ens_pre_info['pre_neurons'],P['n_syn']))
		locations[inpt]=np.zeros((ens_pre_info['pre_neurons'],P['n_syn']))
		for pre in range(ens_pre_info['pre_neurons']):
			for syn in range(P['n_syn']):
				locations[inpt][pre][syn]=P['hyperopt'][inpt][pre][syn]['l']
				weights[inpt][pre][syn]=P['hyperopt'][inpt][pre][syn]['w']
	return weights,locations,bias

def create_bioneuron(P,weights,locations,bias):
	bioneuron=Bahl(P,bias)
	bioneuron.make_synapses(P,weights,locations)
	bioneuron.start_recording()
	return bioneuron	

def run_bioneuron_events(P,bioneuron,spikes_ideal,signals_spikes_pres):
	neuron.h.dt = P['dt_neuron']*1000
	neuron.init()
	inpts=[key for key in signals_spikes_pres.iterkeys()]
	pres=[signals_spikes_pres[inpt]['spikes_in'].shape[1] for inpt in inpts]
	all_input_spikes=[signals_spikes_pres[inpt]['spikes_in'] for inpt in inpts]
	for time in range(spikes_ideal.shape[0]):
		t_neuron=time*P['dt_nengo']*1000
		for i in range(len(inpts)):  #for each input connection
			for pre in range(pres[i]): #for each input neuron
				if all_input_spikes[i][time][pre] > 0: #if input neuron spikes at time
					bioneuron.event_step(t_neuron,inpts[i],pre)
		neuron.run(time*P['dt_nengo']*1000)

def filter_spikes(P,bioneuron,spikes_ideal):
	lpf=nengo.Lowpass(P['kernel']['tau'])
	timesteps=np.arange(0,P['optimize']['t_final'],P['dt_nengo'])
	#convert spike times to a spike train for bioneuron spikes
	#todo: cleanup and check
	spikes_bio=np.zeros_like(timesteps)
	spikes_times_bio=np.array(bioneuron.spikes).ravel()
	st=spikes_times_bio/P['dt_nengo']/1000
	st_int=np.round(st,decimals=1).astype(int) #this leads to slight discrepancies with nengo spike trains
	for idx in st_int:
		if idx >= len(spikes_bio): break
		spikes_bio[idx]=1.0/P['dt_nengo']
	spikes_bio=spikes_bio.T
	spikes_ideal=spikes_ideal
	rates_bio=lpf.filt(spikes_bio,dt=P['dt_nengo'])
	rates_ideal=lpf.filt(spikes_ideal,dt=P['dt_nengo'])
	return spikes_bio,spikes_ideal,rates_bio,rates_ideal

def compute_loss(P,rates_bio,rates_ideal):
	loss=np.sqrt(np.average((rates_bio-rates_ideal)**2))
	return loss

def export_data(P,weights,locations,bias,signals_spikes_pres,spikes_bio,spikes_ideal,rates_bio,rates_ideal):
	try:
		os.makedirs('eval_%s_bioneuron_%s'%(P['eval'],P['hyperopt']['bionrn']))
		os.chdir('eval_%s_bioneuron_%s'%(P['eval'],P['hyperopt']['bionrn']))
	except OSError:
		os.chdir('eval_%s_bioneuron_%s'%(P['eval'],P['hyperopt']['bionrn']))
	np.savez('bias.npz',bias=bias)
	np.savez('spikes_rates_bio_ideal.npz',
				spikes_bio=spikes_bio,spikes_ideal=spikes_ideal,
				rates_bio=rates_bio,rates_ideal=rates_ideal)
	for inpt in signals_spikes_pres.iterkeys():
		np.savez('%s_weights.npz'%inpt,weights=weights[inpt])
		np.savez('%s_locations.npz'%inpt,locations=locations[inpt])
	os.chdir('..')

def plot_spikes_rates(P,best_results_file,target_signal):
	spikes_bio=[]
	spikes_ideal=[]
	rates_bio=[]
	rates_ideal=[]
	for file in best_results_file:
		spikes_rates_bio_ideal=np.load(file+'/spikes_rates_bio_ideal.npz')
		spikes_bio.append(spikes_rates_bio_ideal['spikes_bio'])
		spikes_ideal.append(spikes_rates_bio_ideal['spikes_ideal'])
		rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
		rates_ideal.append(spikes_rates_bio_ideal['rates_ideal'])
	spikes_bio=np.array(spikes_bio).T
	spikes_ideal=np.array(spikes_ideal).T
	rates_bio=np.array(rates_bio).T
	rates_ideal=np.array(rates_ideal).T
	loss=np.sqrt(np.average((rates_bio-rates_ideal)**2))
	sns.set(context='poster')
	figure1, (ax0,ax1,ax2) = plt.subplots(3, 1,sharex=True)
	timesteps=np.arange(0,P['optimize']['t_final'],P['dt_nengo'])
	ax0.plot(timesteps,target_signal)
	rasterplot(timesteps,spikes_ideal,ax=ax1,use_eventplot=True)
	rasterplot(timesteps,spikes_bio,ax=ax2,use_eventplot=True)
	ax0.set(ylabel='input signal \n(weighted sum)',title='total rmse (rate)=%.5f'%loss)
	ax1.set(ylabel='ideal spikes')
	ax2.set(ylabel='bio spikes')
	figure1.savefig('spikes_bio_vs_ideal.png')
	plt.close()
	for nrn in range(rates_bio.shape[1]):
		figure,ax=plt.subplots(1,1)
		bio_rates_plot=ax.plot(timesteps,rates_bio[:,nrn][:len(timesteps)],linestyle='-')
		ideal_rates_plot=ax.plot(timesteps,rates_ideal[:,nrn][:len(timesteps)],linestyle='--',
			color=bio_rates_plot[0].get_color())
		ax.plot(0,0,color='k',linestyle='-',label='bioneuron')
		ax.plot(0,0,color='k',linestyle='--',label='LIF')
		rmse=np.sqrt(np.average((rates_bio[:,nrn][:len(timesteps)]-rates_ideal[:,nrn][:len(timesteps)])**2))
		ax.set(xlabel='time (s)',ylabel='firing rate (Hz)',title='rmse=%.5f'%rmse)
		figure.savefig('bio_vs_ideal_rates_neuron_%s'%nrn)
		plt.close(figure)

def plot_hyperopt_loss(P,losses):
	import pandas as pd
	columns=('bioneuron','eval','loss')
	df=pd.DataFrame(columns=columns,index=np.arange(0,losses.shape[0]*losses.shape[1]))
	i=0
	for bionrn in range(losses.shape[0]):
		for hyp_eval in range(losses.shape[1]):
			df.loc[i]=[bionrn,hyp_eval,losses[bionrn][hyp_eval]]
			i+=1
	sns.set(context='poster')
	figure1,ax1=plt.subplots(1,1)
	sns.tsplot(time="eval",value="loss",unit='bioneuron',data=df)
	ax1.set(xlabel='trial',ylabel='loss')
	figure1.savefig('total_hyperopt_performance.png')
	plt.close(figure1)	











def simulate(P_hyperopt):
	P=P_hyperopt
	os.chdir(P['directory']+P['ens_post']['atrb']['label'])
	#load in pre signsls, pre spikes, ideal spikes for this bioneuron
	signals_spikes_pres,spikes_ideal=load_signals_spikes(P)
	spikes_ideal=spikes_ideal[:,P['hyperopt']['bionrn']]
	#load in hyperopt biases, locations, weights
	weights,locations,bias=load_hyperopt_weights_onebyone(P)
	#create bioneuron
	# print 'bionrn',P['hyperopt']['bionrn']	
	# print 'bias',bias
	# print 'weight',weights['pre'][-1][-1]
	# print 'location',locations['pre'][-1][-1]
	bioneuron=create_bioneuron(P,weights,locations,bias)
	#run bioneurons using event-based spike delivery every timestep
	run_bioneuron_events(P,bioneuron,spikes_ideal,signals_spikes_pres)
	# print 'spikes',np.array(bioneuron.spikes)
	#filter ideal and bio spikes to get rates
	spikes_bio,spikes_ideal,rates_bio,rates_ideal=filter_spikes(P,bioneuron,spikes_ideal)
	#compute loss
	loss=compute_loss(P,rates_bio,rates_ideal)
	#save and export spikes, rates of ideal, bio
	export_data(P,weights,locations,bias,signals_spikes_pres,spikes_bio,spikes_ideal,rates_bio,rates_ideal)
	return {'loss': loss, 'eval':P['eval'], 'status': hyperopt.STATUS_OK} 

def run_hyperopt(P_hyperopt):
	#try loading hyperopt trials object from a previous run to pick up where it left off
	os.chdir(P_hyperopt['directory']+P_hyperopt['ens_post']['atrb']['label'])
	try:
		trials=pickle.load(open('bioneuron_%s_hyperopt_trials.p'%P_hyperopt['hyperopt']['bionrn'],'rb'))
		hyp_evals=np.arange(len(trials),P_hyperopt['evals'])
	except IOError:
		trials=hyperopt.Trials()
		hyp_evals=range(P_hyperopt['evals'])
	for t in hyp_evals:
		P_hyperopt['eval']=t
		my_seed=P_hyperopt['hyperopt_seed']+P_hyperopt['ens_post']['atrb']['seed']+P_hyperopt['hyperopt']['bionrn']*(t+1)
		best=hyperopt.fmin(simulate,
			rstate=np.random.RandomState(seed=my_seed),
			space=P_hyperopt,
			algo=hyperopt.tpe.suggest,
			max_evals=(t+1),
			trials=trials)
		print 'Connections into %s, bioneuron %s, hyperopt %s%%'\
			%(P_hyperopt['ens_post']['atrb']['label'],P_hyperopt['hyperopt']['bionrn'],100.0*(t+1)/P_hyperopt['evals'])
	#find best run's directory location
	losses=[t['result']['loss'] for t in trials]
	ids=[t['result']['eval'] for t in trials]
	idx=np.argmin(losses)
	loss=np.min(losses)
	result=str(ids[idx])
	#plot hyperopt performance
	# sns.set(context='poster')
	# figure1,ax1=plt.subplots(1,1)
	# ax1.plot(range(len(trials)),losses)
	# ax1.set(xlabel='trial',ylabel='total loss')
	# figure1.savefig('bioneuron_%s_hyperopt_performance.png'%P_hyperopt['hyperopt']['bionrn'])
	# plt.close(figure1)
	#save trials object for continued optimization later
	pickle.dump(trials,open('bioneuron_%s_hyperopt_trials.p'%P_hyperopt['hyperopt']['bionrn'],'wb'))
	#returns eval number with minimum loss for this bioneuron
	return [P_hyperopt['hyperopt']['bionrn'],int(result),losses] #added losses

def load_bioneuron_system(P_in):
	P=copy.copy(P_in)
	P['ens_post']={}
	P['ens_post']['inpts']=P_in['ens_post'].neuron_type.father_op.inputs
	P['ens_post']['atrb']=P_in['ens_post'].neuron_type.father_op.ens_atributes
	print 'Loading connections into %s' %P['ens_post']['atrb']['label']
	os.chdir(P['directory']+P['ens_post']['atrb']['label'])
	rates_bio=[]
	best_results_file=np.load('best_results_file.npz')['best_results_file']
	for file in best_results_file:
		spikes_rates_bio_ideal=np.load(file+'/spikes_rates_bio_ideal.npz')
		rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
	rates_bio=np.array(rates_bio).T
	target_signal=np.load('target_signal.npz')['target_signal']
	return best_results_file,target_signal,rates_bio

def optimize_bioneuron_system(P_in,my_P):
	P=copy.copy(P_in)
	P['ens_post']={}
	P['ens_post']['inpts']=P_in['ens_post'].neuron_type.father_op.inputs
	P['ens_post']['atrb']=P_in['ens_post'].neuron_type.father_op.ens_atributes
	print 'Optimizing connections into %s' %P['ens_post']['atrb']['label']
	try:
		os.makedirs(P['directory']+P['ens_post']['atrb']['label'])
		os.chdir(P['directory']+P['ens_post']['atrb']['label'])
	except OSError:
		os.chdir(P['directory']+P['ens_post']['atrb']['label'])
	#make and save spikes from pre and ideal populations
	make_pre_and_ideal_spikes(P)
	#run hyperopt to optimize biases, synaptic weights, (and locations)
	P_list=[]
	pool = Pool(nodes=P['n_processes'])
	rng=np.random.RandomState(seed=P['hyperopt_seed']+P['ens_post']['atrb']['seed'])
	for bionrn in range(P['ens_post']['atrb']['neurons']):
		P_hyperopt=make_hyperopt_search_space_onebyone(P,my_P,bionrn,rng)
		# run_hyperopt(P_hyperopt)
		P_list.append(P_hyperopt)
	results=pool.map(run_hyperopt,P_list)
	#save a list of the evals associated with the minimum loss for each bioneuron [bioneuron,eval]
	best_results_file=[]
	rates_bio=[]
	losses=[]
	for bionrn in range(len(results)):
		best_results_file.append(P['directory']+P['ens_post']['atrb']['label']+
				'/eval_%s_bioneuron_%s'%(results[bionrn][1],bionrn))
		spikes_rates_bio_ideal=np.load(best_results_file[-1]+'/spikes_rates_bio_ideal.npz')
		rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
		losses.append(results[bionrn][2])
	rates_bio=np.array(rates_bio).T
	os.chdir(P['directory']+P['ens_post']['atrb']['label'])
	target_signal=np.load('target_signal.npz')['target_signal']
	#plot the spikes and rates of the best run
	plot_spikes_rates(P,best_results_file,target_signal)
	plot_hyperopt_loss(P,np.array(losses))
	np.savez('best_results_file.npz',best_results_file=best_results_file)
	return best_results_file,target_signal,rates_bio