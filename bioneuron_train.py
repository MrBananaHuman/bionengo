#!/usr/bin/env python
'''
/home/psipeter/.virtualenv/test/bin:
/opt/sharcnet/python/2.7.8/intel/bin:
'''
import numpy as np
import nengo
import neuron
import hyperopt
import timeit
import json
import copy
import ipdb
import os
import stat
import sys
import gc
import pickle
from synapses import ExpSyn
import subprocess
from bioneuron_helper import ch_dir, make_signal, load_spikes, load_values, filter_spikes,\
		filter_spikes_2, export_data, plot_spikes_rates_voltage_train,\
		plot_hyperopt_loss, delete_extra_hyperparam_files

class Bahl():
	def __init__(self,P):
		neuron.h.load_file('/home/psipeter/bionengo/NEURON_models/bahl.hoc')
		self.cell = neuron.h.Bahl()
		self.synapses={}
		self.netcons={}
	def add_bias(self,bias):
		self.bias = bias
		self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
		self.bias_current.delay = 0
		self.bias_current.dur = 1e9  # TODO; limits simulation time
		self.bias_current.amp = self.bias
	def make_synapses(self,P,my_weights,my_locations):
		for inpt in P['inpts'].iterkeys():
			self.synapses[inpt]=np.empty((P['inpts'][inpt]['pre_neurons'],P['atrb']['n_syn']),dtype=object)
			for pre in range(P['inpts'][inpt]['pre_neurons']):
				for syn in range(P['atrb']['n_syn']):
					section=self.cell.apical(my_locations[inpt][pre][syn])
					weight=my_weights[inpt][pre][syn]
					synapse=ExpSyn(section,weight,P['atrb']['tau'])
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


def make_hyperopt_space(P_in,bionrn,rng):
	#adds a hyperopt-distributed weight, location, bias for each synapse for each bioneuron,
	#where each neuron is a seperate choice in hyperopt search space
	P=copy.copy(P_in)
	hyperparams={}
	hyperparams['bionrn']=bionrn
	if P['optimize_bias']==True: hyperparams['bias']=hyperopt.hp.uniform('b_%s'%bionrn,-P['b_0'],P['b_0'])
	else: hyperparams['bias']=None
	for inpt in P['inpts'].iterkeys():
		hyperparams[inpt]={}
		for pre in range(P['inpts'][inpt]['pre_neurons']):
			hyperparams[inpt][pre]={}
			for syn in range(P['atrb']['n_syn']):
				hyperparams[inpt][pre][syn]={}
				hyperparams[inpt][pre][syn]['l']=np.round(rng.uniform(0.0,1.0),decimals=2)
				k_distance=2.0 #weight_rescale(hyperparams[inpt][pre][syn]['l'])
				k_neurons=50.0/P['inpts'][inpt]['pre_neurons']
				k_max_rates=300.0/np.average([P['inpts'][inpt]['pre_min_rate'],P['inpts'][inpt]['pre_max_rate']])
				k=k_distance*k_neurons*k_max_rates
				hyperparams[inpt][pre][syn]['w']=hyperopt.hp.uniform('w_%s_%s_%s_%s'%(bionrn,inpt,pre,syn),-k*P['w_0'],k*P['w_0'])
	P['hyperopt']=hyperparams
	return P

def make_hyperopt_space_decomposed_weights(P_in,bionrn,rng):
	P=copy.copy(P_in)
	hyperparams={}
	hyperparams['bionrn']=bionrn
	if P['optimize_bias']==True: hyperparams['bias']=hyperopt.hp.uniform('b_%s'%bionrn,P['bias_min'],P['bias_max'])
	else: hyperparams['bias']=None
	for inpt in P['inpts'].iterkeys():
		decoders=np.load('decoders_from_%s_to_%s.npz'%(inpt,P['atrb']['label']))['decoders']
		k_decoder=1.0/np.linalg.norm(decoders)
		hyperparams[inpt]={}
		for pre in range(P['inpts'][inpt]['pre_neurons']):
			hyperparams[inpt][pre]={}
			hyperparams[inpt][pre]['d']=decoders[pre][0]
			for syn in range(P['atrb']['n_syn']):
				hyperparams[inpt][pre][syn]={}
				hyperparams[inpt][pre][syn]['l']=np.round(rng.uniform(0.0,1.0),decimals=2)
				hyperparams[inpt][pre][syn]['e']=[]
				k_distance=2.0 #weight_rescale(hyperparams[inpt][pre][syn]['l'])
				k_neurons=50.0/P['inpts'][inpt]['pre_neurons']
				k_max_rates=300.0/np.average([P['inpts'][inpt]['pre_min_rate'],P['inpts'][inpt]['pre_max_rate']])
				# k_decoder=1.0/np.average(np.abs(decoders))
				k=k_distance*k_neurons*k_max_rates*k_decoder
				for dim in range(decoders.shape[1]):
					hyperparams[inpt][pre][syn]['e'].append(
						hyperopt.hp.uniform('e_%s_%s_%s_%s_%s'%(bionrn,inpt,pre,syn,dim),-k*P['e_0'],k*P['e_0']))
	P['hyperopt']=hyperparams
	return P

def make_hyperopt_space_decomposed_weights_single_encoder(P_in,bionrn,rng):
	P=copy.copy(P_in)
	hyperparams={}
	hyperparams['bionrn']=bionrn
	if P['optimize_bias']==True: hyperparams['bias']=hyperopt.hp.uniform('b_%s'%bionrn,P['bias_min'],P['bias_max'])
	else: hyperparams['bias']=None
	for inpt in P['inpts'].iterkeys():
		decoders=np.load('decoders_from_%s_to_%s.npz'%(inpt,P['atrb']['label']))['decoders']
		encoders=np.load('encoders_for_%s.npz'%P['atrb']['label'])['encoders'][bionrn]
		k_decoder=1.0/np.linalg.norm(decoders)
		hyperparams[inpt]={}
		hyperparams[inpt]['encoder']=encoders		
		for pre in range(P['inpts'][inpt]['pre_neurons']):
			hyperparams[inpt][pre]={}
			hyperparams[inpt][pre]['d']=decoders[pre][0]
			for syn in range(P['atrb']['n_syn']):
				hyperparams[inpt][pre][syn]={}
				hyperparams[inpt][pre][syn]['l']=np.round(rng.uniform(0.0,1.0),decimals=2)
				hyperparams[inpt][pre][syn]['z']=[]
				k_distance=2.0 #weight_rescale(hyperparams[inpt][pre][syn]['l'])
				k_neurons=50.0/P['inpts'][inpt]['pre_neurons']
				k_max_rates=300.0/np.average([P['inpts'][inpt]['pre_min_rate'],P['inpts'][inpt]['pre_max_rate']])
				k=k_distance*k_neurons*k_max_rates*k_decoder
				for dim in range(decoders.shape[1]):
					hyperparams[inpt][pre][syn]['z'].append(
						hyperopt.hp.uniform('z_%s_%s_%s_%s_%s'%(bionrn,inpt,pre,syn,dim),0.0,k*P['e_0']))
	P['hyperopt']=hyperparams
	return P		

def load_hyperopt_space(P):
	weights={}
	locations={}
	bias=P['hyperopt']['bias']
	for inpt in P['inpts'].iterkeys():
		weights[inpt]=np.zeros((P['inpts'][inpt]['pre_neurons'],P['atrb']['n_syn']))
		locations[inpt]=np.zeros((P['inpts'][inpt]['pre_neurons'],P['atrb']['n_syn']))
		for pre in range(P['inpts'][inpt]['pre_neurons']):
			for syn in range(P['atrb']['n_syn']):
				locations[inpt][pre][syn]=P['hyperopt'][inpt][pre][syn]['l']
				if P['decompose_weights']== True:
					if P['single_encoder']==True: #w_ij=z_i*(d_i dot e_j)
						weights[inpt][pre][syn]=P['hyperopt'][inpt][pre][syn]['z']*np.dot( 
								P['hyperopt'][inpt][pre]['d'],
								np.array(P['hyperopt'][inpt]['encoder']))
					else: #w_ij=d_i dot e_ij
						weights[inpt][pre][syn]=np.dot( 
								P['hyperopt'][inpt][pre]['d'],
								np.array(P['hyperopt'][inpt][pre][syn]['e']))
				else:
					weights[inpt][pre][syn]=P['hyperopt'][inpt][pre][syn]['w']
	return weights,locations,bias

def create_bioneuron(P,weights,locations,bias):
	bioneuron=Bahl(P)
	if P['optimize_bias']==True: bioneuron.add_bias(bias)
	bioneuron.make_synapses(P,weights,locations)
	bioneuron.start_recording()
	return bioneuron	

def compute_loss(P,spikes_bio, spikes_ideal, rates_bio,rates_ideal,voltages):
	rmse=np.sqrt(np.average((rates_bio-rates_ideal)**2))
	if P['complex_loss']==True:
		t_total=len(voltages)
		t_saturated=len(np.where((-40.0<voltages) & (voltages<-20.0))[0]) #when neurons burst initially then saturate, they settle around -40<V<-20
		L_saturated=np.exp(10*t_saturated/t_total) #time spend in saturated regime exponentially increases loss (max e^10)
		#L_startup2=5*np.sqrt(np.average((rates_bio[:100]-rates_ideal[:100])**2))
		# print L_startup2
		# t_startup=0
		# for t in range(30): #first 50 timesteps
		# 	# print t, spikes_bio[t], spikes_ideal[t]
		# 	if spikes_bio[t] != spikes_ideal[t]: t_startup+=1
		# L_startup=10*t_startup
		# print L_startup
		loss=rmse+L_saturated#+L_startup2
	else:
		loss=rmse
	return loss

'''###############################################################################################################'''
'''###############################################################################################################'''

def run_bioneuron_event_based(P,bioneuron,all_spikes_pre):
	neuron.h.dt = P['dt_neuron']*1000
	inpts=[key for key in all_spikes_pre.iterkeys()]
	pres=[all_spikes_pre[inpt].shape[1] for inpt in inpts]
	all_input_spikes=[all_spikes_pre[inpt] for inpt in inpts]
	t_final=all_spikes_pre[inpts[0]].shape[0] #number of timesteps total
	neuron.init()
	# neuron.run(0.1*t_final*P['dt_nengo']*1000) #run out transients (no inputs to model, lets voltages stabilize)
	# bioneuron.start_recording() #reset recording attributes in neuron
	# neuron.init()
	# for time in range(t_final): #for each timestep
	for time in np.arange(1,t_final): #for each timestep
		t_neuron=time*P['dt_nengo']*1000
		# print time, t_neuron
		neuron.run(t_neuron)
		for i in range(len(inpts)):  #for each input connection
			# if inpts[i] == P['atrb']['label'] and time < 50: continue #don't make recurrent connection rely on itself to get going
			for pre in range(pres[i]): #for each input neuron
				if all_input_spikes[i][time][pre] > 0: #if input neuron spikes at time
					bioneuron.event_step(t_neuron,inpts[i],pre)


def simulate(P):
	os.chdir(P['directory']+P['atrb']['label'])
	all_spikes_pre,all_spikes_ideal=load_spikes(P)
	spikes_ideal=all_spikes_ideal[:,P['hyperopt']['bionrn']]
	weights,locations,bias=load_hyperopt_space(P)
	bioneuron=create_bioneuron(P,weights,locations,bias)
	run_bioneuron_event_based(P,bioneuron,all_spikes_pre)
	spikes_bio,spikes_ideal,rates_bio,rates_ideal,voltages=filter_spikes_2(P,bioneuron,spikes_ideal)
	loss=compute_loss(P,spikes_bio, spikes_ideal, rates_bio,rates_ideal,voltages)
	export_data(P,weights,locations,bias,spikes_bio,spikes_ideal,rates_bio,rates_ideal,voltages,loss)
	return {'loss': loss, 'eval':P['current_eval'], 'status': hyperopt.STATUS_OK}

def run_hyperopt(P):
	#try loading hyperopt trials object from a previous run to pick up where it left off
	os.chdir(P['directory']+P['atrb']['label'])
	try:
		trials=pickle.load(open('bioneuron_%s_hyperopt_trials.p'%P['hyperopt']['bionrn'],'rb'))
		hyp_evals=np.arange(len(trials),P['atrb']['evals'])
	except IOError:
		trials=hyperopt.Trials()
		hyp_evals=range(P['atrb']['evals'])
	for t in hyp_evals:
		P['current_eval']=t
		my_seed=P['hyperopt_seed']+P['atrb']['seed']+P['hyperopt']['bionrn']*(t+1)
		best=hyperopt.fmin(simulate,
				rstate=np.random.RandomState(seed=my_seed),
				space=P,
				algo=hyperopt.tpe.suggest,
				max_evals=(t+1),
				trials=trials)
		print 'Connections into %s, bioneuron %s, hyperopt %s%%'\
			%(P['atrb']['label'],P['hyperopt']['bionrn'],100.0*(t+1)/P['atrb']['evals'])
		#save trials object for checkpoints / continued training later
		if t % int(P['atrb']['evals']/3) == 0 and P['save_hyperopt_trials'] == True:
			pickle.dump(trials,open('bioneuron_%s_hyperopt_trials.p'%P['hyperopt']['bionrn'],'wb'))
	#find best run's directory location
	losses=[t['result']['loss'] for t in trials]
	ids=[t['result']['eval'] for t in trials]
	idx=np.argmin(losses)
	loss=np.min(losses)
	result=str(ids[idx])
	pickle.dump(trials,open('bioneuron_%s_hyperopt_trials.p'%P['hyperopt']['bionrn'],'wb'))
	#returns eval number with minimum loss for this bioneuron
	return [P['hyperopt']['bionrn'],int(result),losses]

def train_hyperparams(P):
	print 'Training connections into %s' %P['atrb']['label']
	os.chdir(P['directory']+P['atrb']['label']) #should be created in pre_build_func()
	P_list=[]
	pool = Pool(nodes=P['n_nodes'])
	rng=np.random.RandomState(seed=P['hyperopt_seed']+P['atrb']['seed'])
	for bionrn in range(P['atrb']['neurons']):
		if P['decompose_weights']==True:
			if P['single_encoder']==True: P_hyperopt=make_hyperopt_space_decomposed_weights_single_encoder(P,bionrn,rng)
			else: P_hyperopt=make_hyperopt_space_decomposed_weights(P,bionrn,rng)
		else: P_hyperopt=make_hyperopt_space(P,bionrn,rng)
		# run_hyperopt(P_hyperopt)
		P_list.append(P_hyperopt)
	results=pool.map(run_hyperopt,P_list)
	# pool.terminate()
	#create and save a list of the eval_number associated with the minimum loss for each bioneuron
	best_hyperparam_files, rates_bio, best_losses, all_losses = [], [], [], []
	for bionrn in range(len(results)):
		best_hyperparam_files.append(P['directory']+P['atrb']['label']+'/eval_%s_bioneuron_%s'%(results[bionrn][1],bionrn))
		spikes_rates_bio_ideal=np.load(best_hyperparam_files[-1]+'/spikes_rates_bio_ideal.npz')
		best_losses.append(np.load(best_hyperparam_files[-1]+'/loss.npz')['loss'])
		rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
		all_losses.append(results[bionrn][2])
	rates_bio=np.array(rates_bio).T
	os.chdir(P['directory']+P['atrb']['label'])
	target=np.load('output_ideal_%s.npz'%P['atrb']['label'])['values']
	#delete files not in best_hyperparam_files
	delete_extra_hyperparam_files(P,best_hyperparam_files)
	#plot the spikes and rates of the best run
	plot_spikes_rates_voltage_train(P,best_hyperparam_files,target,np.array(best_losses))
	plot_hyperopt_loss(P,np.array(all_losses))
	np.savez('best_hyperparam_files.npz',best_hyperparam_files=best_hyperparam_files)
	return best_hyperparam_files,target,rates_bio

'''###############################################################################################################'''
'''###############################################################################################################'''

def train_hyperparams_serial_farming(P):
	print 'Training connections into %s' %P['atrb']['label']
	os.chdir(P['directory']+P['atrb']['label']) #should be created in pre_build_func()
	param_name='params_ensemble_%s.json'%P['atrb']['label']
	with open(param_name,'w') as file:
		json.dump(P,file)
	#id_filenames=['idfile_bioneuron_%s.txt'%bionrn for bionrn in range(P['atrb']['neurons'])]
	#make a bash script to submit training for each neuron
	bashfilename='/home/psipeter/.local/bin/bash/bash_submit_train.sh'
	bashfile=open(bashfilename,'w')
	# bashfile.write('#!/bin/bash/\n')
	bashfile.write('DEST_DIR=%s\n'%(P['directory']+P['atrb']['label']))
	bashfile.write('PARAMNAME=%s\n'%param_name)
	bashfile.write('N_NEURONS=%s\n'%(P['atrb']['neurons']-1))
	bashfile.write('RUNTIME=%s\n'%P['runtime'])
	bashfile.write('MEMORY=%s\n'%P['memory'])
	bashfile.write('declare -a ID_FILESNAMES\n')
	#for bionrn in range(P['atrb']['neurons']):
		#bashfile.write('ID_FILENAMES[%s]=%s'%(bionrn,id_filenames[bionrn]))
	bashfile.write('cd ${DEST_DIR}\n')
	bashfile.write('for bionrn in `seq 0 ${N_NEURONS}`; do\n')
	bashfile.write('\techo "Submitting bioneuron training: ${bionrn} of ${N_NEURONS}"\n')
	bashfile.write('\tOUTFILE="output_bioneuron_${bionrn}.txt"\n')
	bashfile.write('\tIDFILE="jobid_bioneuron_${bionrn}.txt"\n')
	bashfile.write('\tsqsub -r ${RUNTIME} -q serial -o ${OUTFILE} --idfile=${IDFILE} --mpp=${MEMORY} /home/psipeter/bionengo/sqsub_train.py  ${bionrn} ${PARAMNAME}\n')
	bashfile.write('\tdone;\n')
	bashfile.close()
	st = os.stat(bashfilename)
	os.chmod(bashfilename, st.st_mode | stat.S_IEXEC)
	subprocess.call(bashfilename,shell=True)
	sqjobs_list=[]
	for bionrn in range(P['atrb']['neurons']):
		with open('jobid_bioneuron_%s.txt'%bionrn,'r') as file:
			jobid=int(file.read().splitlines()[0])
			sqjobs_list.append(jobid)

	#make a bash script to collect the output files from the first sqjobs
	#only submit when previous sqjobs have finished
	sqjobs_string=[','.join(str(x) for x in sqjobs_list)][0]
	bashfilename='/home/psipeter/.local/bin/bash/bash_collect_train.sh'
	bashfile=open(bashfilename,'w')
	bashfile.write('DEST_DIR=%s\n'%(P['directory']+P['atrb']['label']))
        bashfile.write('PARAMNAME=%s\n'%param_name)
       # bashfile.write('N_NEURONS=%s\n'%P['atrb']['neurons'])
        bashfile.write('RUNTIME=%s\n'%P['runtime'])
        bashfile.write('MEMORY=%s\n'%P['memory'])
        #bashfile.write('SQJOBS=%s\n'%sqjobs_list)
	bashfile.write('cd ${DEST_DIR}\n')
        bashfile.write('echo "Submitting collection job, waiting for training..."\n')
        bashfile.write('OUTFILE="output_collection.txt"\n')
        bashfile.write('sqsub -r ${RUNTIME} -q serial -o ${OUTFILE} -w %s --mpp=${MEMORY} /home/psipeter/bionengo/sqsub_collect.py ${PARAMNAME}\n'%sqjobs_string)
        bashfile.write('done;\n')
        bashfile.close()
        st = os.stat(bashfilename)
        os.chmod(bashfilename, st.st_mode | stat.S_IEXEC)
        subprocess.call(bashfilename,shell=True)
	return
