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
		if seed == None:
			seed=np.random.randint(99999999)
	if signal_type=='prime_sinusoids':
		raw_signal=signals.prime_sinusoids(dt,t_final,dim)
		return raw_signal
	raw_signal=[]
	for d in range(dim):
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

def make_pre_and_ideal_spikes(P,ens_post):
	atrb=ens_post.neuron_type.father_op.ens_atributes
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
	with nengo.Network() as opt_model:
		ideal=nengo.Ensemble(
				label=atrb['label'],
				n_neurons=atrb['neurons'],
				dimensions=atrb['dim'],
				max_rates=nengo.dists.Uniform(atrb['min_rate'],atrb['max_rate']),
				radius=atrb['radius'],
				seed=atrb['seed'])
		p_ideal_spikes=nengo.Probe(ideal.neurons,'spikes')
		for key in ens_post.neuron_type.father_op.inputs.iterkeys():
			pre_atrb=ens_post.neuron_type.father_op.inputs[key]
			signals[key]=make_signal(P['optimize'])
			stims[key]=nengo.Node(lambda t, n=key: signals[key][:,np.floor(t/P['dt_nengo'])])
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
		opt_test.run(P['optimize']['t_final'])
	for key in ens_post.neuron_type.father_op.inputs.iterkeys():
		signal_in=opt_test.data[p_pres[key]]
		spikes_in=opt_test.data[p_pres_spikes[key]]
		np.savez(P['directory']+'signals_spikes_from_%s_to_%s.npz'%(key,ens_post.label),
						signal_in=signal_in,spikes_in=spikes_in)
	spikes_ideal=opt_test.data[p_ideal_spikes]
	np.savez(P['directory']+'ideal_spikes_%s.npz'%ens_post.label,spikes_ideal=spikes_ideal)















# def weight_rescale(location):
# 	#interpolation
# 	import numpy as np
# 	from scipy.interpolate import interp1d
# 	#load voltage attenuation data for bahl.hoc
# 	# voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
# 	voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
# 	f_voltage_att = interp1d(voltage_attenuation['distances'],voltage_attenuation['voltages'])
# 	scaled_weight=1.0/f_voltage_att(location)
# 	return scaled_weight

def make_hyperopt_search_space(P_in,ens_post):
	#adds a hyperopt-distributed weight, location, bias for each synapse for each bioneuron
	#scale max weight to statistics of conn.pre ensemble and synapse location:
	#farther along dendrite means (2x) higher max weights allowed
	#fewer neurons and lower max_rates means higher max weights allowed
	#todo: make explict scaling functions for these relationships
	P=copy.copy(P_in)
	P['weights']={}
	P['locations']={}
	P['biases']={}
	for bionrn in range(ens_post.n_neurons):
		P['biases'][bionrn]=hyperopt.hp.uniform('b_%s'%bionrn,P['bias_min'],P['bias_max'])
		for inpt in ens_post.neuron_type.father_op.inputs.iterkeys():
			ens_pre_info=ens_post.neuron_type.father_op.inputs[inpt]
			k_distance=2.0 #weight_rescale(P['locations']['%s_%s_%s_%s'%(bionrn,inpt,pre,syn)])
			k_neurons=50.0/ens_pre_info['pre_neurons']
			k_max_rates=300.0/np.average([ens_pre_info['pre_min_rate'],ens_pre_info['pre_max_rate']])
			k=k_distance*k_neurons*k_max_rates
			for pre in range(ens_pre_info['pre_neurons']):
				#todo: draw an integer from a normal distribution to calculate synapse number
				#todo: implement weight_rescale for weights given number drawn from hp.uniform
				for syn in range(P['n_syn']):
					if P['optimize_locations']==True:
						P['locations']['%s_%s_%s_%s'%(bionrn,inpt,pre,syn)]=\
							hyperopt.hp.uniform('l_%s_%s_%s_%s'%(bionrn,inpt,pre,syn),0.0,1.0)
					else:
						P['locations']['%s_%s_%s_%s'%(bionrn,inpt,pre,syn)]=np.random.uniform(0.0,1.0)
					P['weights']['%s_%s_%s_%s'%(bionrn,inpt,pre,syn)]=\
						hyperopt.hp.uniform('w_%s_%s_%s_%s'%(bionrn,inpt,pre,syn),-k*P['w_0'],k*P['w_0'])
	return P

def make_hyperopt_search_space_choice(P_in,ens_post):
	#adds a hyperopt-distributed weight, location, bias for each synapse for each bioneuron,
	#where each neuron is a seperate choice in hyperopt search space
	P=copy.copy(P_in)
	cases=[]
	for bionrn in range(ens_post.n_neurons):
		cases.append({})
		cases[-1]['bionrn']=bionrn
		cases[-1]['bias']=hyperopt.hp.uniform('b_%s'%bionrn,P['bias_min'],P['bias_max'])
		for inpt in ens_post.neuron_type.father_op.inputs.iterkeys():
			cases[-1][inpt]={}
			ens_pre_info=ens_post.neuron_type.father_op.inputs[inpt]
			k_distance=2.0 #weight_rescale(P['locations']['%s_%s_%s_%s'%(bionrn,inpt,pre,syn)])
			k_neurons=50.0/ens_pre_info['pre_neurons']
			k_max_rates=300.0/np.average([ens_pre_info['pre_min_rate'],ens_pre_info['pre_max_rate']])
			k=k_distance*k_neurons*k_max_rates
			for pre in range(ens_pre_info['pre_neurons']):
				cases[-1][inpt][pre]={}
				for syn in range(P['n_syn']):
					cases[-1][inpt][pre][syn]={}
					if P['optimize_locations']==True:
						cases[-1][inpt][pre][syn]['l']=hyperopt.hp.uniform('l_%s_%s_%s_%s'%(bionrn,inpt,pre,syn),0.0,1.0)
					else:
						cases[-1][inpt][pre][syn]['l']=np.random.uniform(0.0,1.0)
					cases[-1][inpt][pre][syn]['w']=hyperopt.hp.uniform('w_%s_%s_%s_%s'%(bionrn,inpt,pre,syn),-k*P['w_0'],k*P['w_0'])
	# P['hyperopt']=hyperopt.hp.choice('bionrn',cases)
	P['hyperopt']=cases
	return P

def load_signals_spikes(P,ens_post):
	signals_spikes_pres={}
	for key in ens_post.neuron_type.father_op.inputs.iterkeys():
		signals_spikes_pres[key]=np.load(P['directory']+'signals_spikes_from_%s_to_%s.npz'%(key,ens_post.label))
	spikes_ideal=np.load(P['directory']+'ideal_spikes_%s.npz'%ens_post.label)['spikes_ideal']
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
	def make_synapses(self,P,ens_post,bionrn,my_weights,my_locations):
		for inpt in ens_post.neuron_type.father_op.inputs.iterkeys():
			ens_pre_info=ens_post.neuron_type.father_op.inputs[inpt]
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

def load_hyperopt_weights(P,ens_post):
	weights={}
	locations={}
	biases={}
	for bionrn in range(ens_post.n_neurons):
		biases[bionrn]=P['biases'][bionrn]
		weights[bionrn]={}
		locations[bionrn]={}
		for inpt in ens_post.neuron_type.father_op.inputs.iterkeys():
			ens_pre_info=ens_post.neuron_type.father_op.inputs[inpt]
			weights[bionrn][inpt]=np.zeros((ens_pre_info['pre_neurons'],P['n_syn']))
			locations[bionrn][inpt]=np.zeros((ens_pre_info['pre_neurons'],P['n_syn']))
			for pre in range(ens_pre_info['pre_neurons']):
				for syn in range(P['n_syn']):
					locations[bionrn][inpt][pre][syn]=P['locations']['%s_%s_%s_%s'%(bionrn,inpt,pre,syn)]
					weights[bionrn][inpt][pre][syn]=P['weights']['%s_%s_%s_%s'%(bionrn,inpt,pre,syn)]
	return weights,locations,biases

def load_hyperopt_weights_choice(P,ens_post):
	weights={}
	locations={}
	biases={}
	for bionrn in range(ens_post.n_neurons):
		biases[bionrn]=P['hyperopt'][bionrn]['bias']
		weights[bionrn]={}
		locations[bionrn]={}
		for inpt in ens_post.neuron_type.father_op.inputs.iterkeys():
			ens_pre_info=ens_post.neuron_type.father_op.inputs[inpt]
			weights[bionrn][inpt]=np.zeros((ens_pre_info['pre_neurons'],P['n_syn']))
			locations[bionrn][inpt]=np.zeros((ens_pre_info['pre_neurons'],P['n_syn']))
			for pre in range(ens_pre_info['pre_neurons']):
				for syn in range(P['n_syn']):
					locations[bionrn][inpt][pre][syn]=P['hyperopt'][bionrn][inpt][pre][syn]['l']
					weights[bionrn][inpt][pre][syn]=P['hyperopt'][bionrn][inpt][pre][syn]['w']
	return weights,locations,biases

def create_biopop(P,ens_post,weights,locations,biases):
	biopop={}
	for bionrn in range(ens_post.n_neurons):
		bioneuron=Bahl(P,biases[bionrn])
		bioneuron.make_synapses(P,ens_post,bionrn,weights[bionrn],locations[bionrn])
		bioneuron.start_recording()
		biopop[bionrn]=bioneuron
	return biopop	

def run_biopop_events(P,ens_post,biopop,spikes_ideal,signals_spikes_pres):
	neuron.h.dt = P['dt_neuron']*1000
	neuron.init()
	inpts=[key for key in signals_spikes_pres.iterkeys()]
	pres=[signals_spikes_pres[inpt]['spikes_in'].shape[1] for inpt in inpts]
	all_input_spikes=[signals_spikes_pres[inpt]['spikes_in'] for inpt in inpts]
	for time in range(spikes_ideal.shape[0]):
		t_neuron=time*P['dt_nengo']*1000
		for bioneuron in biopop.itervalues(): #for each bioneuron
			for i in range(len(inpts)):  #for each input connection
				for pre in range(pres[i]): #for each input neuron
					if all_input_spikes[i][time][pre] > 0: #if input neuron spikes at time
						bioneuron.event_step(t_neuron,inpts[i],pre)
		neuron.run(time*P['dt_nengo']*1000)

def filter_spikes(P,biopop,spikes_ideal):
	lpf=nengo.Lowpass(P['kernel']['tau'])
	timesteps=np.arange(0,P['optimize']['t_final'],P['dt_nengo'])
	#convert spike times to a spike train for bioneuron spikes
	#todo: cleanup and check
	spikes_bio=[]
	for bioneuron in biopop.itervalues():
		my_spikes_bio=np.zeros_like(timesteps)
		spikes_times_bio=np.array(bioneuron.spikes).ravel()
		st=spikes_times_bio/P['dt_nengo']/1000
		st_int=np.round(st,decimals=1).astype(int)
		for idx in st_int:
			if idx >= len(my_spikes_bio): break
			my_spikes_bio[idx]=1.0/P['dt_nengo']
		spikes_bio.append(my_spikes_bio)
	spikes_bio=np.array(spikes_bio).T
	spikes_ideal=spikes_ideal
	rates_bio=np.array([lpf.filt(spikes_bio[:,nrn],dt=P['dt_nengo']) for nrn in range(spikes_bio.shape[1])]).T
	rates_ideal=np.array([lpf.filt(spikes_ideal[:,nrn],dt=P['dt_nengo']) for nrn in range(spikes_ideal.shape[1])]).T
	return spikes_bio,spikes_ideal,rates_bio,rates_ideal

def compute_loss(P,rates_bio,rates_ideal):
	losses=[]
	for nrn in range(rates_bio.shape[1]):
		my_loss=np.sqrt(np.average((rates_bio[:,nrn]-rates_ideal[:,nrn])**2))
		losses.append(my_loss)
	total_loss=np.sum(losses)
	return total_loss

def export_data(P,ens_post,result_file,weights,locations,biases,signals_spikes_pres,
				spikes_bio,spikes_ideal,rates_bio,rates_ideal):
	os.makedirs(result_file)
	os.chdir(result_file)
	np.savez('biases.npz',biases=biases)
	np.savez('spikes_rates_bio_ideal.npz',
				spikes_bio=spikes_bio,spikes_ideal=spikes_ideal,
				rates_bio=rates_bio,rates_ideal=rates_ideal)
	probe_signals=[]
	for key in signals_spikes_pres.iterkeys():
		probe_signals.append(signals_spikes_pres[key]['signal_in'])
	target_signal=np.sum(probe_signals,axis=0)
	np.savez('target_signal.npz',target_signal=target_signal)
	for bionrn in range(ens_post.n_neurons):
		os.makedirs(str(bionrn))
		os.chdir(str(bionrn))
		for inpt in signals_spikes_pres.iterkeys():
			os.makedirs(str(inpt))
			os.chdir(str(inpt))
			np.savez('weights.npz',weights=weights[bionrn][inpt])
			np.savez('locations.npz',locations=weights[bionrn][inpt])
			os.chdir('..')
		os.chdir('..')

def plot_spikes_rates(P,best_result_file,loss):
	target_signal=np.load(best_result_file+'/target_signal.npz')['target_signal']
	spikes_rates_bio_ideal=np.load(best_result_file+'/spikes_rates_bio_ideal.npz')
	spikes_bio=spikes_rates_bio_ideal['spikes_bio']
	spikes_ideal=spikes_rates_bio_ideal['spikes_ideal']
	rates_bio=spikes_rates_bio_ideal['rates_bio']
	rates_ideal=spikes_rates_bio_ideal['rates_ideal']
	sns.set(context='poster')
	figure1, (ax0,ax1,ax2) = plt.subplots(3, 1,sharex=True)
	timesteps=np.arange(0,P['optimize']['t_final'],P['dt_nengo'])
	ax0.plot(timesteps,target_signal)
	rasterplot(timesteps,spikes_ideal,ax=ax1,use_eventplot=True)
	rasterplot(timesteps,spikes_bio,ax=ax2,use_eventplot=True)
	ax0.set(ylabel='input signal \n(weighted sum)',title='total rmse (rate)=%.5f'%loss)
	ax1.set(ylabel='ideal spikes',title='total rmse (rate)=%.5f'%loss)
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













def simulate(P):
	ens_post=P['ens_post']
	result_file=P['directory']+ens_post.label+'/'+make_addon(6)
	#load in pre signsls, pre spikes, ideal spikes
	# start_load=timeit.default_timer()
	signals_spikes_pres,spikes_ideal=load_signals_spikes(P,ens_post)
	# stop_load=timeit.default_timer()
	# print 'load_signals_spikes_time=%.4f'%(stop_load-start_load)
	#load in hyperopt biases, locations, weights
	# start_weights=timeit.default_timer()
	# weights,locations,biases=load_hyperopt_weights(P,ens_post)
	weights,locations,biases=load_hyperopt_weights_choice(P,ens_post)
	# stop_weights=timeit.default_timer()
	# print 'load_weights_time=%.4f'%(stop_weights-start_weights)
	#create bioneurons
	# start_biopop=timeit.default_timer()
	biopop=create_biopop(P,ens_post,weights,locations,biases)
	# stop_biopop=timeit.default_timer()
	# print 'create_biopop_time=%.4f'%(stop_biopop-start_biopop)
	#run bioneurons using event-based spike delivery every timestep
	# start_run=timeit.default_timer()
	run_biopop_events(P,ens_post,biopop,spikes_ideal,signals_spikes_pres)
	# stop_run=timeit.default_timer()
	# print 'run_biopop_time=%.4f'%(stop_run-start_run)	
	#filter ideal and bio spikes to get rates
	spikes_bio,spikes_ideal,rates_bio,rates_ideal=filter_spikes(P,biopop,spikes_ideal)
	#compute loss
	loss=compute_loss(P,rates_bio,rates_ideal)
	#save and export spikes, rates of ideal, bio
	export_data(P,ens_post,result_file,weights,locations,biases,signals_spikes_pres,
				spikes_bio,spikes_ideal,rates_bio,rates_ideal)
	return {'loss': loss, 'result_file':result_file, 'status': hyperopt.STATUS_OK} 



def optimize_bioneuron_system(P_in):
	ens_post=P_in['ens_post']
	print 'Optimizing connections into %s' %ens_post.label
	P=copy.copy(P_in)
	os.makedirs(P['directory']+ens_post.label)
	os.chdir(P['directory']+ens_post.label)
	# start_input_spikes=timeit.default_timer()
	make_pre_and_ideal_spikes(P,ens_post)
	# stop_input_spikes=timeit.default_timer()
	# print 'make_intput_spikes_time=%.4f'%(stop_input_spikes-start_input_spikes)
	#run hyperopt to optimize biases, synaptic weights, (and locations)
	# start_search_space=timeit.default_timer()
	# P_hyperopt=make_hyperopt_search_space(P,ens_post)
	P_hyperopt=make_hyperopt_search_space_choice(P,ens_post)
	# stop_search_space=timeit.default_timer()
	# print 'make_search_space_time=%.4f'%(stop_search_space-start_search_space)

	trials=hyperopt.Trials()
	for t in range(P['evals']):
		best=hyperopt.fmin(simulate,
			space=P_hyperopt,
			algo=hyperopt.tpe.suggest,
			max_evals=(t+1),
			trials=trials)
		print 'connections into %s; hyperopt %s%% complete'%(ens_post.label,100.0*(t+1)/P['evals'])
	os.chdir('..')

	#find best run's directory location
	losses=[t['result']['loss'] for t in trials]
	ids=[t['result']['result_file'] for t in trials]
	idx=np.argmin(losses)
	loss=np.min(losses)
	best_result_file=str(ids[idx])

	#plot hyperopt performance
	sns.set(context='poster')
	figure1,ax1=plt.subplots(1,1)
	losses=[t['result']['loss'] for t in trials]
	ax1.plot(range(len(trials)),losses)
	ax1.set(xlabel='trial',ylabel='total loss')
	figure1.savefig('hyperopt_performance.png')

	#plot the spikes and rates of the best run
	plot_spikes_rates(P,best_result_file,loss)

	with open('best_result_file.txt','wb') as outfile:
		json.dump(best_result_file,outfile)
	spikes_rates_bio_ideal=np.load(best_result_file+'/spikes_rates_bio_ideal.npz')
	rates_bio=spikes_rates_bio_ideal['rates_bio']
	target_signal=np.load(best_result_file+'/target_signal.npz')['target_signal']
	os.chdir(P['directory'])
	return best_result_file,target_signal,rates_bio