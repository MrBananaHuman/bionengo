import string
import random
import signals
import numpy as np
import os
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot

def make_pre_ideal_spikes(P,network):
	bio_dict={}
	#opt_net=copy.copy(network)
	for ens in opt_net.ensembles:
		if not isinstance(ens.neuron_type,BahlNeuron): continue #only bioensembles selected
		ens.neuron_type=nengo.LIF() #will this work?
		bio_dict[ens.label]={}
		with opt_net: bio_dict[ens.label]['probe']=nengo.Probe(ens,synapse=ens.synapse)
		bio_dict[ens.label]['inputs']={}
		for conn in opt_net.connections:
			if isinstance(conn.post_obj,BahlNeuron):
				bio_dict[ens.label]['inputs'][conn.pre_obj.label]={}
				with opt_net: 
					bio_dict[ens.label]['inputs'][conn.pre_obj.label]['probe']=\
							nengo.Probe(conn.pre_obj.neurons,'spikes')
	#rebuild network?
	#define input signals and connect to inputs
	with nengo.Simulator(opt_net,dt=P['dt_nengo']) as opt_sim:
		opt_sim.run(P['train']['t_final'])
	for bio in bio_dict.iterkeys():
		try: 
			os.makedirs(bio)
			os.chdir(bio)
		except OSError:
			os.chdir(bio)
		bio_dict[bio]['ideal_spikes']=opt_sim.data[bio_dict[bio]['probe']]
		for inpt in bio_dict[bio]['inputs'].iterkeys():
			bio_dict[bio]['inputs'][inpt]['pre_spikes']=\
					opt_sim.data[bio_dict[bio]['inputs'][inpt]['probe']]
			np.savez('spikes_from_%s_to_%s.npz'%(inpt,bio),spikes=\
					bio_dict[bio]['inputs'][inpt]['pre_spikes'])
		np.savez('spikes_ideal_%s.npz'%bio,spikes=bio_dict[bio]['ideal_spikes'])
		os.chdir('..')

def ch_dir():
	#change directory for data and plot outputs
	root=os.getcwd()
	def make_addon(N):
		addon=str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(N)))
		return addon
	addon=make_addon(9)
	datadir=root+'/data/'+addon+'/' #linux or mac
	os.makedirs(datadir)
	os.chdir(datadir) 
	return datadir

def make_signal(P):
	""" Returns: array indexed by t when called from a nengo Node"""
	signal_type=P['type']
	dt=P['dt']
	t_final=P['t_final']+dt #why?
	dim=P['dim']
	raw_signal=[]
	for d in range(dim):
		if signal_type=='sinusoid':
			raw_signal.append(signals.sinusoid(dt,t_final,P['omega']))
		if signal_type=='constant':
			raw_signal.append(signals.constant(dt,t_final,P['mean']))
		elif signal_type=='white':
			raw_signal.append(signals.white(dt,t_final))
		elif signal_type=='white_binary':
			raw_signal.append(signals.white_binary(dt,t_final,P['mean'],P['std']))
		elif signal_type=='switch':
			raw_signal.append(signals.switch(dt,t_final,P['max_freq'],))
		elif signal_type=='equalpower':
			if seed == None: seed=np.random.randint(99999999)
			raw_signal.append(signals.equalpower(dt,t_final,P['max_freq'],mean=P['mean'],std=P['std'],seed=seed))
		elif signal_type=='poisson_binary':
			raw_signal.append(signals.poisson_binary(dt,t_final,mean_freq,P['max_freq'],low,high))
		elif signal_type=='poisson':
			raw_signal.append(signals.poisson(dt,t_final,mean_freq,P['max_freq']))
		elif signal_type=='pink_noise':
			raw_signal.append(signals.pink_noise(dt,t_final,P['mean'],P['std']))
	assert len(raw_signal) > 0, "signal type not specified"
	return np.array(raw_signal)

def weight_rescale(location):
	from scipy.interpolate import interp1d
	#load voltage attenuation data for bahl.hoc
	voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
	f_voltage_att = interp1d(voltage_attenuation['distances'],voltage_attenuation['voltages'])
	scaled_weight=1.0/f_voltage_att(location)
	return scaled_weight

def load_spikes(P):
	all_spikes_pre={}
	for key in P['inpts'].iterkeys():
		all_spikes_pre[key]=np.load('spikes_from_%s_to_%s.npz'%(key,P['atrb']['label']))['spikes']
	spikes_ideal=np.load('spikes_ideal_%s.npz'%P['atrb']['label'])['spikes']
	return all_spikes_pre,spikes_ideal

def filter_spikes(P,bioneuron,spikes_ideal):
	lpf=nengo.Lowpass(P['kernel']['tau'])
	timesteps=np.arange(0,P['train']['t_final'],P['dt_nengo'])
	#convert spike times to a spike train for bioneuron spikes
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

def export_data(P,weights,locations,bias,spikes_bio,spikes_ideal,rates_bio,rates_ideal):
	try:
		os.makedirs('eval_%s_bioneuron_%s'%(P['current_eval'],P['hyperopt']['bionrn']))
		os.chdir('eval_%s_bioneuron_%s'%(P['current_eval'],P['hyperopt']['bionrn']))
	except OSError:
		os.chdir('eval_%s_bioneuron_%s'%(P['current_eval'],P['hyperopt']['bionrn']))
	np.savez('bias.npz',bias=bias)
	#spikes_ideal and rates_ideal are redundant (saved in pre_build_func()), but makes loading easier
	np.savez('spikes_rates_bio_ideal.npz',
				spikes_bio=spikes_bio,spikes_ideal=spikes_ideal,
				rates_bio=rates_bio,rates_ideal=rates_ideal)
	for inpt in P['inpts'].iterkeys():
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
	timesteps=np.arange(0,P['train']['t_final'],P['dt_nengo'])
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

def load_hyperparams(P_in):
	P=copy.copy(P_in)
	P['ens']={}
	P['ens']['inpts']=P_in['ens_post'].neuron_type.father_op.inputs
	P['ens']['atrb']=P_in['ens_post'].neuron_type.father_op.ens_atributes
	print 'Loading connections into %s' %P['ens']['atrb']['label']
	os.chdir(P['directory']+P['ens']['atrb']['label'])
	rates_bio=[]
	best_hyperparam_files=np.load('best_hyperparam_files.npz')['best_hyperparam_files']
	for file in best_hyperparam_files:
		spikes_rates_bio_ideal=np.load(file+'/spikes_rates_bio_ideal.npz')
		rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
	rates_bio=np.array(rates_bio).T
	target_signal=np.load('target_signal.npz')['target_signal']
	return best_hyperparam_files,target_signal,rates_bio