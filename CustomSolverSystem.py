import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal
import copy
import ipdb
import json
import os
from optimize_bioneuron_system_onebyone import load_bioneuron_system, optimize_bioneuron_system,make_signal,ch_dir
from synapses import ExpSyn, Exp2Syn
from tqdm import *
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot

class CustomSolver(nengo.solvers.Solver):
	def __init__(self,P,ens_post,model,method):
		self.P=P
		self.ens_post=ens_post
		self.method=method
		self.weights=False #decoders not weights
		self.bahl_op=None
		self.activities=None
		self.target=None
		self.decoders=None
		self.solver=nengo.solvers.LstsqL2()

	def __call__(self,A,Y,rng=None,E=None):
		from BahlNeuronSystem import BahlNeuron
		if self.decoders != None:
			return self.decoders, dict()
		elif isinstance(self.ens_post.neuron_type, BahlNeuron):
			self.P['ens_post']=self.ens_post
			P=copy.copy(self.P)
			P['ens_post']={}
			P['ens_post']['inpts']=self.P['ens_post'].neuron_type.father_op.inputs
			P['ens_post']['atrb']=self.P['ens_post'].neuron_type.father_op.ens_atributes
			#don't optimize if filenames already exists
			# if self.ens_post.neuron_type.best_results_file != None:
			# 	#todo: restart a failed optimization run without freezing
			# 	best_results_file, targets, activities = load_bioneuron_system(self.P)
			# else:
			best_results_file, targets, activities = optimize_bioneuron_system(self.P)
			self.ens_post.neuron_type.father_op.best_results_file=best_results_file
			if self.method == 'load':
				#load activities/targets from the runs performed during optimization
				print  'loading target signal and bioneuron activities for %s' %P['ens_post']['atrb']['label']
				self.targets=targets
				self.activities=activities
				self.decoders=self.solver(self.activities,self.targets)[0]
			elif self.method == 'simulate':
				#use loaded weights/locations/biases to simulate new activities/targets
				from optimize_bioneuron_system_onebyone import create_bioneuron
				print  'simulating target signal and bioneuron activities for %s' %P['ens_post']['atrb']['label']
				#generate new signals and input spikes
				target,pre_spikes_list,ideal_spikes=make_pre_spikes(P)
				#load in biases, locations, weights from best_results_files
				biopop=create_biopop(P,self.ens_post.neuron_type.father_op)
				#run bioneurons using event-based spike delivery every timestep
				run_biopop_events(P,biopop,pre_spikes_list)
				#filter ideal and bio spikes to get rates
				bio_spikes,bio_rates,ideal_spikes,ideal_rates=filter_spikes(P,biopop,ideal_spikes)
				plot_spikes_rates(P,target,bio_spikes,bio_rates,ideal_spikes,ideal_rates)
				self.targets=nengo.Lowpass(P['kernel']['tau']).filt(target)
				self.activities=bio_rates
				self.decoders=self.solver(self.activities,self.targets)[0]
				plot_decode(P,self.targets,self.activities,self.decoders)
				del biopop
			# print 'activities', self.activities
			# print 'target', self.targets
			# print 'decoders', self.decoders
			return self.decoders, dict()





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





def make_pre_spikes(P):
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
	if P['decode']['type']=='sinusoid':
		n_inputs=len(P['ens_post']['inpts'])
		primes=sigz.primeno(n_inputs)
		i=0
	with nengo.Network() as decode_model:
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
	with nengo.Simulator(decode_model,dt=P['dt_nengo']) as decoder_test:
		decoder_test.run(P['decode']['t_final'])
	all_signals=[]
	all_pre_spikes=[]
	lpf=nengo.Lowpass(P['kernel']['tau'])
	for key in P['ens_post']['inpts'].iterkeys():
		signal_in=lpf.filt(
					# lpf.filt(
					transforms[key]*decoder_test.data[p_stims[key]],
						# dt=P['dt_nengo']),
					dt=P['dt_nengo'])
		spikes=decoder_test.data[p_pres_spikes[key]]
		all_signals.append(signal_in)
		all_pre_spikes.append(spikes)
	summed_signals=np.sum(all_signals,axis=0)
	all_pre_spikes=np.array(all_pre_spikes)
	ideal_spikes=decoder_test.data[p_ideal_spikes]
	return summed_signals,all_pre_spikes,ideal_spikes


def create_biopop(P,bahl_op):
	biopop=[]
	for bionrn in range(P['ens_post']['atrb']['neurons']):
		bias=np.load(bahl_op.best_results_file[bionrn]+'/bias.npz')['bias']
		bioneuron=Bahl(P,bias)
		for inpt in bahl_op.inputs.iterkeys():
			pre_neurons=bahl_op.inputs[inpt]['pre_neurons']
			pre_synapses=bahl_op.P['n_syn']
			bioneuron.synapses[inpt]=np.empty((pre_neurons,pre_synapses),dtype=object)
			# bioneuron.netcons[inpt]=np.empty((pre_neurons,pre_synapses),dtype=object)	
			weights=np.load(bahl_op.best_results_file[bionrn]+'/'+inpt+'_weights.npz')['weights']
			locations=np.load(bahl_op.best_results_file[bionrn]+'/'+inpt+'_locations.npz')['locations']
			for pre in range(pre_neurons):
				for syn in range(pre_synapses):	
					section=bioneuron.cell.apical(locations[pre][syn])
					weight=weights[pre][syn]
					synapse=ExpSyn(section,weight,bahl_op.P['tau'])
					bioneuron.synapses[inpt][pre][syn]=synapse
		bioneuron.start_recording()
		biopop.append(bioneuron)
	return biopop

def	run_biopop_events(P,biopop,pre_spikes_list):
	print 'Decoder simulation for %s...'%P['ens_post']['atrb']['label']
	neuron.h.dt = P['dt_neuron']*1000
	neuron.init()
	inpts=[key for key in P['ens_post']['inpts'].iterkeys()]
	for time in tqdm(range(pre_spikes_list.shape[1])): #for each simulated nengo timestep
		t_neuron=time*P['dt_nengo']*1000
		for bioneuron in biopop:
			for i in range(pre_spikes_list.shape[0]):  #for each input connection
				for pre in range(pre_spikes_list[i][time].shape[0]): #for each input neuron
					if pre_spikes_list[i][time][pre] > 0: #if input neuron spikes at time
						bioneuron.event_step(t_neuron,inpts[i],pre)
		neuron.run(t_neuron)

def filter_spikes(P,biopop,ideal_spikes):
	lpf=nengo.Lowpass(P['kernel']['tau'])
	timesteps=np.arange(0,P['decode']['t_final'],P['dt_nengo'])
	#convert spike times to a spike train for bioneuron spikes
	all_spikes_bio=[]
	all_rates_bio=[]
	all_rates_ideal=[]
	for bionrn in range(len(biopop)):
		bioneuron=biopop[bionrn]
		spikes_bio=np.zeros_like(timesteps)
		spikes_times_bio=np.array(bioneuron.spikes).ravel()
		st=spikes_times_bio/P['dt_nengo']/1000
		st_int=np.round(st,decimals=1).astype(int)
		for idx in st_int:
			if idx >= len(spikes_bio): break
			spikes_bio[idx]=1.0/P['dt_nengo']
		spikes_bio=spikes_bio.T
		rates_bio=lpf.filt(spikes_bio,dt=P['dt_nengo'])
		rates_ideal=lpf.filt(ideal_spikes[:,bionrn],dt=P['dt_nengo'])
		all_spikes_bio.append(spikes_bio)
		all_rates_bio.append(rates_bio)
		all_rates_ideal.append(rates_ideal)
	bio_spikes=np.array(all_spikes_bio).T
	bio_rates=np.array(all_rates_bio).T
	ideal_rates=np.array(all_rates_ideal).T
	return bio_spikes,bio_rates,ideal_spikes,ideal_rates

def plot_spikes_rates(P,target,bio_spikes,bio_rates,ideal_spikes,ideal_rates):
	loss=np.sqrt(np.average((bio_rates-ideal_rates)**2))
	sns.set(context='poster')
	figure1, (ax0,ax1,ax2) = plt.subplots(3, 1,sharex=True)
	timesteps=np.arange(0,P['decode']['t_final'],P['dt_nengo'])
	ax0.plot(timesteps,target)
	rasterplot(timesteps,ideal_spikes,ax=ax1,use_eventplot=True)
	rasterplot(timesteps,bio_spikes,ax=ax2,use_eventplot=True)
	ax0.set(ylabel='input signal \n(weighted sum)',title='total rmse (rate)=%.5f'%loss)
	ax1.set(ylabel='ideal spikes')
	ax2.set(ylabel='bio spikes')
	figure1.savefig('spikes_bio_vs_ideal_CustomSolver_%s.png'%P['ens_post']['atrb']['label'])
	plt.close()
	# for nrn in range(rates_bio.shape[1]):
	# 	figure,ax=plt.subplots(1,1)
	# 	bio_rates_plot=ax.plot(timesteps,rates_bio[:,nrn][:len(timesteps)],linestyle='-')
	# 	ideal_rates_plot=ax.plot(timesteps,rates_ideal[:,nrn][:len(timesteps)],linestyle='--',
	# 		color=bio_rates_plot[0].get_color())
	# 	ax.plot(0,0,color='k',linestyle='-',label='bioneuron')
	# 	ax.plot(0,0,color='k',linestyle='--',label='LIF')
	# 	rmse=np.sqrt(np.average((rates_bio[:,nrn][:len(timesteps)]-rates_ideal[:,nrn][:len(timesteps)])**2))
	# 	ax.set(xlabel='time (s)',ylabel='firing rate (Hz)',title='rmse=%.5f'%rmse)
	# 	figure.savefig('bio_vs_ideal_rates_neuron_%s'%nrn)
	# 	plt.close(figure)

def plot_decode(P,targets,activities,decoders):
	timesteps=np.arange(0,P['decode']['t_final'],P['dt_nengo'])
	xhat=np.dot(activities,decoders)
	rmse=np.sqrt(np.average((targets-xhat)**2))
	figureA,axA=plt.subplots(1,1)
	axA.plot(timesteps,targets,label='$x(t)$')
	axA.plot(timesteps,xhat,label='$\hat{x}(t)$')
	axA.set(xlabel='time',ylabel='value',title='rmse=%.5f'%rmse)
	axA.legend()
	figureA.savefig('decoder_accuracy_CustomSolver_%s.png'%P['ens_post']['atrb']['label'])
	plt.close()