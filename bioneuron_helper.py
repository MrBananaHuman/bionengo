import string
import random
import signals
import numpy as np
import os
import nengo
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
import ipdb

def ch_dir(P):
	#change directory for data and plot outputs
	if P['platform'] == 'workstation': root=os.getcwd()
	elif P['platform'] == 'sharcnet': root='/work/psipeter/bionengo'
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
			if P['seed'] == None: seed=np.random.randint(99999999)
			else: seed=P['seed']
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

def load_inputs_ideal(P):
	all_spikes_pre={}
	for key in P['inpts'].iterkeys():
		all_spikes_pre[key]=np.load('spikes_from_%s_to_%s.npz'%(key,P['atrb']['label']))['spikes']
	input_current_ideal=np.load('input_current_ideal_%s.npz'%P['atrb']['label'])['input_current_ideal']
	spikes_ideal=np.load('spikes_ideal_%s.npz'%P['atrb']['label'])['spikes']
	return all_spikes_pre,spikes_ideal,input_current_ideal

def load_values(P):
	#loads decoded outputs of all pre ensembles
	all_values_pre={}
	for key in P['inpts'].iterkeys():
		all_values_pre[key]=np.load('output_from_%s_to_%s.npz'%(key,P['atrb']['label']))['values']
	spikes_ideal=np.load('spikes_ideal_%s.npz'%P['atrb']['label'])['spikes']
	return all_values_pre,spikes_ideal

def filter_spikes(P,bioneuron,spikes_ideal):
	lpf=nengo.Lowpass(P['kernel']['tau'])
	timesteps=np.arange(0,P['train']['t_final'],P['dt_nengo'])
	#convert spike times to a spike train for bioneuron spikes
	spikes_bio=np.zeros_like(timesteps)
	spikes_times_bio=np.array(bioneuron.spikes).ravel()
	st=spikes_times_bio/P['dt_nengo']/1000
	st_int=np.round(st,decimals=0).astype(int) #this leads to slight discrepancies with nengo spike trains
	# st_int=np.round(st,decimals=1).astype(int) #this leads to slight discrepancies with nengo spike trains
	for idx in st_int:
		if idx >= len(spikes_bio): break
		spikes_bio[idx]=1.0/P['dt_nengo']
	spikes_bio=spikes_bio.T
	spikes_ideal=spikes_ideal
	rates_bio=lpf.filt(spikes_bio,dt=P['dt_nengo'])
	rates_ideal=lpf.filt(spikes_ideal,dt=P['dt_nengo'])
	voltages=np.array(bioneuron.v_record).ravel()
	return spikes_bio,spikes_ideal,rates_bio,rates_ideal,voltages

def post_process_arrays(P,bioneuron,spikes_ideal,currents_ideal):
	#filters spikes and downsamples currents so that loss can be computed on rates and/or currents
	#make as similar to the make_step method in bioneuron_builder as possible
	lpf=nengo.Lowpass(P['kernel']['tau'])
	timesteps=np.arange(0,P['train']['t_final'],P['dt_nengo'])
	#convert spike times to a spike train for bioneuron spikes
	spikes_bio=[]
	spikes_times_bio=np.array(bioneuron.spikes)
	for t in timesteps:
		spikes_current=len(np.where(spikes_times_bio<1000*t)[0])
		spikes_last=np.count_nonzero(spikes_bio)
		count=spikes_current-spikes_last
		spikes_bio.append(1.0*count/P['dt_nengo'])
		# print spikes_current, spikes_last
	# ipdb.set_trace()
	spikes_bio=np.array(spikes_bio)
	spikes_ideal=spikes_ideal
	rates_bio=lpf.filt(spikes_bio,dt=P['dt_nengo'])
	rates_ideal=lpf.filt(spikes_ideal,dt=P['dt_nengo'])

	currents_bio=(np.array(bioneuron.v_apical_end)-np.array(bioneuron.v_soma_begin)) / bioneuron.ri_apical #(v_dend - v_soma)/R_dend
	# currents_bio=np.array(bioneuron.v_apical_end) / bioneuron.ri_apical #v_dend/R_dend
	
	#downsample the currents_bio vector to account for differences in dt_nengo vs dt_neuron
	ratio=int(P['dt_nengo']/P['dt_neuron'])
	currents_bio_filled=np.zeros((ratio*currents_ideal.shape[0]))
	currents_bio_filled[:currents_bio.shape[0]]=currents_bio
	current_bio_reshaped=currents_bio_filled.reshape(-1,ratio)
	downsampled_currents_bio=np.mean(current_bio_reshaped,axis=1)
	voltages=np.array(bioneuron.v_record).ravel()

	return spikes_bio,spikes_ideal,rates_bio,rates_ideal,downsampled_currents_bio,voltages

def export_data(P,weights,locations,bias,spikes_bio,spikes_ideal,
				rates_bio,rates_ideal,currents_bio,currents_ideal,voltages,loss):
	try: 
		os.chdir('bioneuron_%s'%P['hyperopt']['bionrn'])
		current_best_loss=np.load('loss.npz')['loss']
		if loss < current_best_loss: #new best evaluation
			np.savez('bias.npz',bias=bias)
			np.savez('loss.npz',loss=loss)
			#spikes_ideal and rates_ideal are redundant (saved in pre_build_func()), but makes loading easier
			np.savez('spikes_rates_bio_ideal.npz',
						spikes_bio=spikes_bio,spikes_ideal=spikes_ideal,
						rates_bio=rates_bio,rates_ideal=rates_ideal,
						currents_bio=currents_bio,currents_ideal=currents_ideal,
						voltages=voltages)
			for inpt in P['inpts'].iterkeys():
				np.savez('%s_weights.npz'%inpt,weights=weights[inpt])
				np.savez('%s_locations.npz'%inpt,locations=locations[inpt])
	except OSError: #first evaluation
		os.makedirs('bioneuron_%s'%P['hyperopt']['bionrn'])
		os.chdir('bioneuron_%s'%P['hyperopt']['bionrn'])
		np.savez('bias.npz',bias=bias)
		np.savez('loss.npz',loss=loss)
		#spikes_ideal and rates_ideal are redundant (saved in pre_build_func()), but makes loading easier
		np.savez('spikes_rates_bio_ideal.npz',
					spikes_bio=spikes_bio,spikes_ideal=spikes_ideal,
					rates_bio=rates_bio,rates_ideal=rates_ideal,
					currents_bio=currents_bio,currents_ideal=currents_ideal,
					voltages=voltages)
		for inpt in P['inpts'].iterkeys():
			np.savez('%s_weights.npz'%inpt,weights=weights[inpt])
			np.savez('%s_locations.npz'%inpt,locations=locations[inpt])
	os.chdir('..')
	# try:
	# 	os.makedirs('eval_%s_bioneuron_%s'%(P['current_eval'],P['hyperopt']['bionrn']))
	# 	os.chdir('eval_%s_bioneuron_%s'%(P['current_eval'],P['hyperopt']['bionrn']))
	# except OSError:
	# 	os.chdir('eval_%s_bioneuron_%s'%(P['current_eval'],P['hyperopt']['bionrn']))


# def delete_extra_hyperparam_files(P,best_hyperparam_files):
# 	from os import listdir
# 	from os.path import isdir, join
# 	import shutil
# 	onlydirs = [join(P['directory']+P['atrb']['label'],f) for f in listdir(P['directory']+P['atrb']['label']) if isdir(join(P['directory']+P['atrb']['label'], f))]
# 	for f in onlydirs:
# 		if f not in best_hyperparam_files: shutil.rmtree(f)

def make_plots(P,best_hyperparam_files,target_signal,losses,all_losses):
	import pandas as pd
	spikes_bio=[]
	spikes_ideal=[]
	rates_bio=[]
	rates_ideal=[]
	currents_bio=[]
	currents_ideal=[]
	voltages=[]
	for filename in best_hyperparam_files:
		if P['platform']=='workstation':
			filename=str(filename).replace('/work','/home')
			filename=filename.replace('/psipeter','/pduggins')
		spikes_rates_bio_ideal=np.load(filename+'/spikes_rates_bio_ideal.npz')
		spikes_bio.append(spikes_rates_bio_ideal['spikes_bio'])
		spikes_ideal.append(spikes_rates_bio_ideal['spikes_ideal'])
		rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
		rates_ideal.append(spikes_rates_bio_ideal['rates_ideal'])
		currents_bio.append(spikes_rates_bio_ideal['currents_bio'])
		currents_ideal.append(spikes_rates_bio_ideal['currents_ideal'])		
		voltages.append(spikes_rates_bio_ideal['voltages'])
	spikes_bio=np.array(spikes_bio).T
	spikes_ideal=np.array(spikes_ideal).T
	rates_bio=np.array(rates_bio).T
	rates_ideal=np.array(rates_ideal).T
	currents_bio=np.array(currents_bio).T
	currents_ideal=np.array(currents_ideal).T	
	voltages=np.array(voltages).T
	rmse_rate=np.sqrt(np.average((rates_bio-rates_ideal)**2))
	rmse_current=np.sqrt(np.average((currents_bio-currents_ideal)**2))

	#spike raster comparison
	sns.set(context='poster')
	figure1, (ax0,ax1,ax2) = plt.subplots(3, 1,sharex=True)
	timesteps=np.arange(0,P['train']['t_final'],P['dt_nengo'])
	ax0.plot(timesteps,target_signal)
	rasterplot(timesteps,spikes_ideal,ax=ax1,use_eventplot=True)
	rasterplot(timesteps,spikes_bio,ax=ax2,use_eventplot=True)
	ax0.set(ylabel='input signal \n(weighted sum)',title='total rmse (rate)=%.5f'%rmse_rate)
	ax1.set(ylabel='ideal spikes')
	ax2.set(ylabel='bio spikes')
	figure1.savefig('spikes_bio_vs_ideal.png')
	plt.close()

	#neuron-by-neuron rate, current, and voltage comparison
	try:
		os.makedirs('bioneuron_plots')
		os.chdir('bioneuron_plots')
	except:
		os.chdir('bioneuron_plots')
	for nrn in range(rates_bio.shape[1]):
		figure,ax=plt.subplots(1,1)
		ax.plot(timesteps,rates_bio[:,nrn][:len(timesteps)],label='bio')
		ax.plot(timesteps,rates_ideal[:,nrn][:len(timesteps)],label='ideal')
		loss=losses[nrn]
		rmse_rate=np.sqrt(np.average((rates_bio[:,nrn][:len(timesteps)]-rates_ideal[:,nrn][:len(timesteps)])**2))
		ax.set(xlabel='time (s)',ylabel='firing rate (Hz)',title='rmse_rate=%.5f, loss=%.5f'%(rmse_rate,loss))
		legend=ax.legend(prop={'size':8})
		figure.savefig('bio_vs_ideal_rates_neuron_%s'%nrn)
		plt.close(figure)

		figure2,ax2=plt.subplots(1,1)
		ax2.plot(timesteps,currents_bio[:,nrn][:len(timesteps)],label='bio')
		ax2.plot(timesteps,currents_ideal[:,nrn][:len(timesteps)],label='ideal')
		loss=losses[nrn]
		rmse_current=np.sqrt(np.average((currents_bio[:,nrn][:len(timesteps)]-currents_ideal[:,nrn][:len(timesteps)])**2))
		legend2=ax2.legend(prop={'size':8})
		ax2.set(xlabel='time (s)',ylabel='current (nA)',title='rmse_current=%.5f, loss=%.5f'%(rmse_current,loss))
		figure2.savefig('bio_vs_ideal_currents_neuron_%s'%nrn)
		plt.close(figure2)

		figure3,ax3=plt.subplots(1,1)
		ax3.plot(voltages[:,nrn])
		ax3.set(xlabel='time (ms)',ylabel='Voltage (mV)')
		figure3.savefig('bioneuron_%s_voltages_train.png'%nrn)
		plt.close(figure3)		
	os.chdir('..')

	# columns=('bioneuron','eval','loss')
	# df=pd.DataFrame(columns=columns,index=np.arange(0,all_losses.shape[0]*all_losses.shape[1]))
	# i=0
	# for bionrn in range(all_losses.shape[0]):
	# 	for hyp_eval in range(all_losses.shape[1]):
	# 		df.loc[i]=[bionrn,hyp_eval,all_losses[bionrn][hyp_eval]]
	# 		i+=1
	# sns.set(context='poster')
	# figure1,ax1=plt.subplots(1,1)
	# sns.tsplot(time="eval",value="loss",unit='bioneuron',data=df)
	# ax1.set(xlabel='trial',ylabel='loss')
	# figure1.savefig('total_hyperopt_performance.png')
	# plt.close(figure1)


def load_hyperparams(P_in):
	P=copy.copy(P_in)
	print 'Loading connections into %s' %P['atrb']['label']
	os.chdir(P['directory']+P['atrb']['label'])
	rates_bio=[]
	best_losses=[]
	best_hyperparam_files=np.load('best_hyperparam_files.npz')['best_hyperparam_files']
	for filename in best_hyperparam_files:
		if P['platform']=='workstation':
			filename=str(filename).replace('/work','/home')
			filename=filename.replace('/psipeter','/pduggins')
		spikes_rates_bio_ideal=np.load(filename+'/spikes_rates_bio_ideal.npz')
		best_losses.append(np.load(filename+'/loss.npz')['loss'])
		rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
	rates_bio=np.array(rates_bio).T
	target_signal=np.load('output_ideal_%s.npz'%P['atrb']['label'])['values']
	all_losses=np.load('all_losses.npz')['losses']
	if P['platform']=='workstation':
		make_plots(P,best_hyperparam_files,target_signal,np.array(best_losses),np.array(all_losses))
		# plot_spikes_rates_voltage_train(P,best_hyperparam_files,target_signal,np.array(best_losses))
		# plot_hyperopt_loss(P,np.array(all_losses))
	return best_hyperparam_files,target_signal,rates_bio
