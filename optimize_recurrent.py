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

def make_signal(P,train_or_test):
	""" Returns: array indexed by t when called from a nengo Node"""
	import signals
	import numpy as np
	dt=P['dt_nengo']
	if train_or_test == 'train':
		sP=P['signal_train']
		t_final=P['t_train']+dt #why is this extra step necessary?
	elif train_or_test == 'test':
		sP=P['signal_test']
		t_final=P['t_test']+dt #why is this extra step necessary?
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
				dt,t_final,sP['max_freq'],sP['mean'],sP['std'],seed=sP['seed']))
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
		pre = nengo.Ensemble(n_neurons=P['actual_ens_pre_neurons'],
								dimensions=P['ens_pre_dim'],
								max_rates=nengo.dists.Uniform(P['ens_pre_min_rate'],
																P['ens_pre_max_rate']),
								seed=P['actual_ens_pre_seed'],radius=P['ens_pre_radius'],)
		ideal = nengo.Ensemble(n_neurons=P['ens_ideal_neurons'],
								dimensions=P['ens_ideal_dim'],
								max_rates=nengo.dists.Uniform(P['ens_ideal_min_rate'],
																P['ens_ideal_max_rate']),
								seed=P['ens_ideal_seed'],radius=P['ens_ideal_radius'],)
		nengo.Connection(signal,pre,synapse=None)
		nengo.Connection(pre,ideal,synapse=P['tau']) #need to multiply by B
		nengo.Connection(ideal,ideal,synapse=P['tau']) #need to multiply by A
		probe_signal = nengo.Probe(signal)
		probe_pre = nengo.Probe(pre.neurons,'spikes')
		probe_ideal = nengo.Probe(ideal.neurons,'spikes')
	with nengo.Simulator(opt_model,dt=P['dt_nengo']) as opt_sim:
		opt_sim.run(P['t_train'])
	gains=opt_sim.data[ideal].gain
	biases=opt_sim.data[ideal].bias
	encoders=opt_sim.data[ideal].encoders
	signal_in=opt_sim.data[probe_signal]
	spikes_feedforward=opt_sim.data[probe_pre]
	spikes_ideal=opt_sim.data[probe_ideal]
	np.savez(P['inputs']+'lifdata.npz',
			signal_in=signal_in,spikes_feedforward=spikes_feedforward,spikes_ideal=spikes_ideal,
			gains=gains, biases=biases, encoders=encoders)

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

def add_search_space(P_old):
	#adds a hyperopt-distributed weight, location, bias for each synapse on each neuron
	import numpy as np
	import hyperopt
	import copy
	P=copy.copy(P_old)
	P['weights']={}
	P['locations']={}
	P['bias']={}
	for m in range(P['ens_ideal_neurons']):
		P['bias']['%s'%m]=hyperopt.hp.uniform('b_%s'%m,P['bias_min'],P['bias_max'])
		for n in range(P['ens_pre_neurons']):
			for i in range(P['n_syn']): 
				P['locations']['%s_%s_%s'%(m,n,i)] =\
					np.round(np.random.uniform(0.0,1.0),decimals=2)
				#scale max weight to statistics of conn.pre ensemble and synapse location:
				#farther along dendrite means higher max weights allowed
				#fewer neurons and lower max_rates means higher max weights allowed
				#todo: make explict scaling functions for these relationships
				k_distance=weight_rescale(P['locations']['%s_%s_%s'%(m,n,i)])
				k_neurons=50.0/P['ens_pre_neurons']
				k_max_rates=300.0/np.average([P['ens_pre_min_rate'],P['ens_pre_max_rate']])
				k=k_distance*k_neurons*k_max_rates
				P['weights']['%s_%s_%s'%(m,n,i)]=hyperopt.hp.uniform(
							'w_%s_%s_%s'%(m,n,i),-k*P['w_0'],k*P['w_0'])
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

def rate_loss(biopop_rates,ideal_rates):
	import numpy as np
	losses=[np.sqrt(np.average((bio_rates-ideal_rates)**2)) for bio_rates in biopop_rates]
	return losses

def compare_bio_ideal_rates(P,filenames): #rewrite for recurrent
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
	spikes_feedforward=np.array(biopop[0]['spikes_in'])
	bio_spikes=np.array([np.array(biopop[b]['bio_spikes']) for b in range(len(biopop))]).T
	ideal_spikes=np.array([np.array(biopop[b]['ideal_spikes']) for b in range(len(biopop))]).T
	bio_rates=np.array([np.array(biopop[b]['bio_rates']) for b in range(len(biopop))])
	ideal_rates=np.array([np.array(biopop[b]['ideal_rates']) for b in range(len(biopop))])
	loss=np.sqrt(np.average((bio_rates-ideal_rates)**2))
	ipdb.set_trace()

	rasterplot(timesteps,spikes_feedforward,ax=ax0,use_eventplot=True)
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


'''
BIONEURON METHODS 
###################################################################################################
'''

# from collections import namedtuple
import neuron
import numpy as np
from synapses import ExpSyn, Exp2Syn
import os

class Bahl():
    
    def __init__(self):
        # neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
        neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
        self.cell = neuron.h.Bahl()
        self.vecstim = {} #index=input neuron, value=VecStim object (input spike times)
        self.ff_synapses = {} #index=input neuron, value=synapses
        self.ff_netcons = {} #index=input neuron, value=NetCon Object (connection b/w VecStim and Syn)
        self.fb_synapses = {} #index=input neuron, value=synapses
        self.fb_netcons = {} #index=input neuron, value=NetCon Object (connection b/w VecStim and Syn)

    def start_recording(self):
        self.v_record = neuron.h.Vector()
        self.v_record.record(self.cell.soma(0.5)._ref_v)
        self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
        self.t_record = neuron.h.Vector()
        self.t_record.record(neuron.h._ref_t)
        self.spikes = neuron.h.Vector()
        self.ap_counter.record(neuron.h.ref(self.spikes))

    def add_bias(self,bias):
        self.bias = bias
        self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
        self.bias_current.delay = 0
        self.bias_current.dur = 1e9  # TODO; limits simulation time
        self.bias_current.amp = self.bias

    def add_ff_connection(self,idx):
        self.ff_synapses[idx]=[] #list of each synapse in this connection
        self.vecstim[idx]={'vstim':[],'vtimes':[]} #list of input spike times from this neuron
        self.ff_netcons[idx]=[] #list of netcon objects between input vecstim and synapses for this nrn

    def add_ff_synapse(self,idx,syn_type,section,weight,tau,tau2=0.005):
        if syn_type == 'ExpSyn':
            self.ff_synapses[idx].append(ExpSyn(section,weight,tau))
        elif syn_type == 'Exp2Syn':
            self.ff_synapses[idx].append(Exp2Syn(section,weight,tau,tau2))

    def add_fb_connection(self,idx):
        self.fb_synapses[idx]=[] #list of each synapse in this connection
        self.fb_netcons[idx]=[] #list of netcon objects between input vecstim and synapses for this nrn     	

    def add_fb_synapse(self,idx,syn_type,section,weight,tau,tau2=0.005):
        if syn_type == 'ExpSyn':
            self.fb_synapses[idx].append(ExpSyn(section,weight,tau))
        elif syn_type == 'Exp2Syn':
            self.fb_synapses[idx].append(Exp2Syn(section,weight,tau,tau2))

def make_biopop(P):
	biopop=[Bahl() for n in range(P['ens_ideal_neurons'])]
	[bioneuron.start_recording() for bioneuron in biopop] #initialize recording attributes
	return biopop

def connect_feedforward(P,biopop,spikes_feedforward):
	import numpy as np
	import neuron
	import ipdb
	biodata=np.load(P['directory']+P['ens_label']+'/pre/biodata.npz') #todo: hardcoded path
	#create synapses to receive input from artificial spiking cells, weights and locations loaded
	for m in range(biodata['weights'].shape[0]): #ideal?
		bioneuron=biopop[m]
		bioneuron.add_bias(biodata['biases'][m]) #todo: not optimized
		for n in range(biodata['weights'].shape[1]): #pre?
			bioneuron.add_ff_connection(n)
			for i in range(biodata['weights'].shape[2]): #synapses?
				syn_type=P['synapse_type']
				section=bioneuron.cell.apical(biodata['locations'][m][n][i])
				weight=biodata['weights'][m][n][i]
				tau=P['tau']
				bioneuron.add_ff_synapse(n,syn_type,section,weight,tau)
	#create spike time vectors and an artificial spiking cell that delivers them
	for m in range(biodata['weights'].shape[0]): #ideal?
		bioneuron=biopop[m]
		for n in range(biodata['weights'].shape[1]): #pre?
			vstim=neuron.h.VecStim()
			bioneuron.vecstim[n]['vstim'].append(vstim)
			spike_times_ms=list(1000*P['dt_nengo']*np.nonzero(spikes_feedforward[:,n])[0]) #timely
			vtimes=neuron.h.Vector(spike_times_ms)
			bioneuron.vecstim[n]['vtimes'].append(vtimes)
			bioneuron.vecstim[n]['vstim'][-1].play(bioneuron.vecstim[n]['vtimes'][-1])
			#connect the VecStim to each synapse
			for syn in bioneuron.ff_synapses[n]:
				netcon=neuron.h.NetCon(bioneuron.vecstim[n]['vstim'][-1],syn.syn)
				netcon.weight[0]=abs(syn.weight)
				bioneuron.ff_netcons[n].append(netcon)

def connect_recurrent(P,biopop,weights,locations):
	import numpy as np
	import neuron
	import ipdb
	for m in range(weights.shape[0]): #ideal?
		bioneuron=biopop[m]
		for n in range(weights.shape[1]): #pre?
			bioneuron.add_fb_connection(n)
			for i in range(weights.shape[2]): #synapse?
				syn_type=P['synapse_type']
				section=bioneuron.cell.apical(locations[m][n][i])
				weight=weights[m][n][i]
				tau=P['tau']
				bioneuron.add_fb_synapse(n,syn_type,section,weight,tau)
				netcon=neuron.h.NetCon(
					biopop[n].cell.soma(0.5)._ref_v,
					bioneuron.fb_synapses[n][i].syn,
					sec=biopop[n].cell.soma)
				biopop[m].fb_netcons[n].append(netcon)

def run_NEURON(P):
	import neuron
	neuron.h.dt = P['dt_neuron']*1000
	neuron.init()
	neuron.run(P['t_train']*1000)

def export_biopop(P,biopop,run_id,weights,locations,signal_in,spikes_feedforward,
					ideal_spikes,biopop_spikes,ideal_rates,biopop_rates,losses):
	import json
	import ipdb
	for m in range(len(biopop)):
		to_export={
			'bio_idx':m,
			'weights':weights[m].tolist(),
			'locations': locations[m].tolist(),
			'bias': biopop[m].bias,
			'signal_in': signal_in.tolist(),
			'spikes_in':spikes_feedforward.tolist(),
			'ideal_spikes': ideal_spikes[m].tolist(),
			'bio_spikes': biopop_spikes[m].tolist(),
			'ideal_rates': ideal_rates[m].tolist(),
			'bio_rates': biopop_rates[m].tolist(),
			'loss': losses[m],
			}
		with open(P['inputs']+run_id+'_bioneuron_%s.json'%m, 'w') as data_file:
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
	import matplotlib.pyplot as plt
	import seaborn
	trials=hyperopt.Trials()
	evals=P['evals_recurrent']
	for t in range(evals): #todo: separate parameter
		best=hyperopt.fmin(simulate,
			space=P,
			algo=hyperopt.tpe.suggest,
			max_evals=(t+1),
			trials=trials)
		print 'Hyperopt %s%%' %(100.0*(t+1)/evals)
	losses=[t['result']['loss'] for t in trials]
	ids=[t['result']['run_id'] for t in trials]
	idx=np.argmin(losses)
	best_run_id=str(ids[idx])
	figure1,ax1=plt.subplots(1,1)
	ax1.plot(range(len(trials)),losses)
	ax1.set(xlabel='trial',ylabel='total loss')
	figure1.savefig('hyperopt_performance.png')
	ipdb.set_trace()
	filename=P['inputs']+best_run_id
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
	spikes_ideal=lifdata['spikes_ideal']
	spikes_feedforward=lifdata['spikes_feedforward']

	weights=np.zeros((P['ens_pre_neurons'],P['ens_ideal_neurons'],P['n_syn']))
	locations=np.zeros((P['ens_pre_neurons'],P['ens_ideal_neurons'],P['n_syn']))
	for m in range(P['ens_ideal_neurons']):
		for n in range(P['ens_pre_neurons']):
			for i in range(P['n_syn']):
				weights[m][n][i]=P['weights']['%s_%s_%s'%(m,n,i)]
				locations[m][n][i]=P['locations']['%s_%s_%s'%(m,n,i)]

	biopop = make_biopop(P)
	connect_feedforward(P,biopop,spikes_feedforward)
	connect_recurrent(P,biopop,weights,locations)

	run_NEURON(P)
	
	biopop_spikes=[get_rates(P,np.round(np.array(nrn.spikes),decimals=3))[0] for nrn in biopop]
	biopop_rates=[get_rates(P,np.round(np.array(nrn.spikes),decimals=3))[1] for nrn in biopop]
	ideal_spikes=[get_rates(P,spikes)[0] for spikes in spikes_ideal.T]
	ideal_rates=[get_rates(P,spikes)[1] for spikes in spikes_ideal.T]

	losses = rate_loss(biopop_rates,ideal_rates)
	loss=np.sum(np.array(losses))

	export_biopop(P,biopop,run_id,weights,locations,signal_in,spikes_feedforward,
					ideal_spikes,biopop_spikes,ideal_rates,biopop_rates,losses)

	for bioneuron in biopop: del bioneuron
	gc.collect()
	stop=timeit.default_timer()
	print loss
	# print 'Simulate Runtime - %s sec' %(stop-start)
	return {'loss': loss, 'run_id':run_id, 'status': hyperopt.STATUS_OK}




def optimize_recurrent(P):
	import json
	import copy
	import ipdb
	import os
	
	P=copy.copy(P)
	P['inputs']=P['directory']+P['ens_label']+'/'+P['ens_pre_label']+'/'
	if os.path.exists(P['inputs']):
		return P['inputs']
	raw_signal=make_signal(P,'train')
	os.makedirs(P['inputs'])
	os.chdir(P['inputs'])
	#todo: hardcoded path
	P['actual_ens_pre_neurons']=50
	P['actual_ens_pre_seed']=333
	generate_ideal_spikes(P,raw_signal)
	P_hyperopt=add_search_space(P)
	filename=run_hyperopt(P_hyperopt)
	filenames=[filename+'_bioneuron_%s.json'%n for n in range(P['ens_ideal_neurons'])]
	with open('filenames.txt','wb') as outfile:
		json.dump(filenames,outfile)
	compare_bio_ideal_rates(P,filenames)
	os.chdir(P['directory'])
	return P['inputs']