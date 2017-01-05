import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal
import copy
import ipdb
import json

class BahlNeuron(nengo.neurons.NeuronType):
	'''compartmental neuron from Bahl et al 2012'''

	probeable=('spikes','voltage')
	def __init__(self,P,inputs=None):
		super(BahlNeuron,self).__init__()
		self.P=P
		self.inputs={} #structured like {ens_in_label: {directory,filenames}}
		if inputs is not None:
			for inpt in inputs:
				with open(inpt['directory']+'filenames.txt','r') as df:
					filenames=json.load(df)
				self.inputs[inpt['label']]={
								'directory':inpt['directory'],
								'filenames':filenames}
		self.optimized=None

	class Bahl():
		def __init__(self):
			neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc') #todo
			# neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc') #todo
			self.cell = neuron.h.Bahl()
			self.synapses = {} #index=input neuron, value=synapses
			self.vecstim = {} #index=input neuron, value=VecStim object (input spike times)
			self.netcons = {} #index=input neuron, value=NetCon Object (connection b/w VecStim and Syn)

		def start_recording(self):
			self.v_record = neuron.h.Vector()
			self.v_record.record(self.cell.soma(0.5)._ref_v)
			self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
			self.t_record = neuron.h.Vector()
			self.t_record.record(neuron.h._ref_t)
			self.spikes = neuron.h.Vector()
			self.ap_counter.record(neuron.h.ref(self.spikes))
			self.nengo_spike_times=[]
			self.nengo_voltages=[]

		def add_bias(self,bias):
			self.bias = bias
			self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
			self.bias_current.delay = 0
			self.bias_current.dur = 1e9 #todo
			self.bias_current.amp = self.bias

		def add_connection(self,idx):
			self.synapses[idx]=[] #list of each synapse in this connection
			#vecstim and netcons only used for optimization, spikes delivered with events for nengo
			self.vecstim[idx]={'vstim':[],'vtimes':[]} #list of input spike times from this neuron
			self.netcons[idx]=[] #list of netcon objects between input vecstim and synapses for this nrn

		def add_synapse(self,idx,syn_type,section,weight,tau,tau2=0.005):
			from synapses import ExpSyn, Exp2Syn
			if syn_type == 'ExpSyn':
				self.synapses[idx].append(ExpSyn(section,weight,tau))
			elif syn_type == 'Exp2Syn':
				self.synapses[idx].append(Exp2Syn(section,weight,tau,tau2))

	def create(self,idx):
		self.bio_idx=idx
		self.bahl=None
		return copy.copy(self)

	def save_optimization(self,conn_pre_label,directory,filenames):
		indices=[]
		weights=[]
		locations=[]
		biases=[]
		signal_in=[]
		bio_spikes=[]
		bio_rates=[]
		ideal_spikes=[]
		ideal_rates=[]
		losses=[]
		for file in filenames: #for each bioneuron
			with open(file,'r') as data_file:
				bioneuron_info=json.load(data_file)
			indices.append(bioneuron_info['bio_idx'])
			weights.append(bioneuron_info['weights'])
			locations.append(bioneuron_info['locations'])
			biases.append(bioneuron_info['bias'])
			signal_in.append(bioneuron_info['signal_in'])
			bio_spikes.append(bioneuron_info['bio_spikes'])
			bio_rates.append(bioneuron_info['bio_rates'])
			ideal_spikes.append(bioneuron_info['ideal_spikes'])
			ideal_rates.append(bioneuron_info['ideal_rates'])
			losses.append(bioneuron_info['loss'])
		self.inputs[conn_pre_label]['directory']=directory
		self.inputs[conn_pre_label]['indices']=np.array(indices)
		self.inputs[conn_pre_label]['weights']=np.array(weights)
		self.inputs[conn_pre_label]['locations']=np.array(locations)
		self.inputs[conn_pre_label]['biases']=np.array(biases)
		self.inputs[conn_pre_label]['signal_in']=np.array(signal_in)[0]
		self.inputs[conn_pre_label]['bio_spikes']=np.array(bio_spikes)
		self.inputs[conn_pre_label]['bio_rates']=np.array(bio_rates)
		self.inputs[conn_pre_label]['ideal_spikes']=np.array(ideal_spikes)
		self.inputs[conn_pre_label]['ideal_rates']=np.array(ideal_rates)
		self.inputs[conn_pre_label]['losses']=np.array(losses)
		np.savez(directory+'/'+'biodata.npz',
				indices=self.inputs[conn_pre_label]['indices'],
				weights=self.inputs[conn_pre_label]['weights'],
				locations=self.inputs[conn_pre_label]['locations'],
				biases=self.inputs[conn_pre_label]['biases'],
				signal_in=self.inputs[conn_pre_label]['signal_in'],
				bio_spikes=self.inputs[conn_pre_label]['bio_spikes'],
				bio_rates=self.inputs[conn_pre_label]['bio_rates'],
				ideal_spikes=self.inputs[conn_pre_label]['ideal_spikes'],
				ideal_rates=self.inputs[conn_pre_label]['ideal_rates'],
				losses=self.inputs[conn_pre_label]['losses'])

	def rates(self, x, gain, bias): #todo: remove this without errors
		return x #CustomSolver class calculates A from Y

	def gain_bias(self, max_rates, intercepts): #todo: remove this without errors
		return np.ones(len(max_rates)),np.ones(len(max_rates))

	def step_math(self,dt,spiked,neurons,voltage,time):
		# if time==dt:
		# 	# neuron.init() #todo: prettier way to initialize first timestep without segfault
		# 	for nrn in neurons:
		# 		nrn.bahl.start_recording()
		desired_t=time*1000
		# ipdb.set_trace()
		'''runs NEURON only for 1st bioensemble in model...OK because transmit_spikes happend already?'''
		neuron.run(desired_t) 
		new_spiked=[]
		new_voltage=[]
		for nrn in neurons:
			spike_times=np.round(np.array(nrn.bahl.spikes),decimals=1)
			count=np.sum(spike_times>(time-dt)*1000)
			new_spiked.append(count)
			nrn.bahl.nengo_spike_times.extend(spike_times[spike_times>(time-dt)*1000])
			volt=np.array(nrn.bahl.v_record)[-1] #fails if neuron.init() not called at right times
			nrn.bahl.nengo_voltages.append(volt)
			new_voltage.append(volt)
			# if len(spike_times)>0: print 'spike time', spike_times[-1], 'voltage', volt
			# else: print 'spike time', 0, 'voltage', volt
		spiked[:]=np.array(new_spiked)/dt
		voltage[:]=np.array(new_voltage)
		# print spiked
		# ipdb.set_trace()

'''
Builder #############################################################################3
'''

class SimBahlNeuron(Operator):
	def __init__(self,neurons,output,voltage,states):
		super(SimBahlNeuron,self).__init__()
		self.neurons=neurons
		self.output=output
		self.voltage=voltage
		self.time=states[0]
		self.reads = [states[0]]
		self.sets=[output,voltage]
		self.updates=[]
		self.incs=[]
		self.neurons.neurons=[self.neurons.create(i) for i in range(output.shape[0])]
		self.P=self.neurons.P
		self.neurons.father_op=self
		neuron.h.dt = self.P['dt_neuron']*1000
		neuron.init()

	def make_step(self,signals,dt,rng):
		output=signals[self.output]
		voltage=signals[self.voltage]
		time=signals[self.time]
		def step_nrn():
			#one bahlneuron object runs step_math, but arg is all cells - what/where returns?
			self.neurons.step_math(dt,output,self.neurons.neurons,voltage,time)
		return step_nrn


class TransmitSpikes(Operator):
	def __init__(self,spikes,bahl_op,states):
		self.spikes=spikes
		self.neurons=bahl_op.neurons.neurons
		self.time=states[0]
		self.reads=[spikes,states[0]]
		self.updates=[]
		self.sets=[]
		self.incs=[]
		neuron.h.dt = bahl_op.P['dt_neuron']*1000
		neuron.init()

	def make_step(self,signals,dt,rng):
		spikes=signals[self.spikes]
		time=signals[self.time]
		def step():
			'''BROKEN: vecstim based method: create spike time vectors 
			and an artificial spiking cell that delivers them'''
			# for nrn in self.neurons: #for each bioneuron
			# 	for i in range(spikes.shape[0]):
			# 		nrn.bahl.vecstim[i]={'vstim':[],'vtimes':[]}
			# 		nrn.bahl.netcons[i]=[]
			# 		vstim=neuron.h.VecStim()
			# 		nrn.bahl.vecstim[i]['vstim'].append(vstim)	

			# for i in range(spikes.shape[0]): #for each input neuron
			# 	if spikes[i] > 0: #if this neuron spiked at this time, then
			# 		spike_time_ms=list([1000*time])
			# 		for nrn in self.neurons: #for each bioneuron
			# 			vtimes=neuron.h.Vector(spike_time_ms)
			# 			nrn.bahl.vecstim[i]['vtimes'].append(vtimes)
			# 			nrn.bahl.vecstim[i]['vstim'][-1].play(nrn.bahl.vecstim[i]['vtimes'][-1])
			# 			for syn in nrn.bahl.synapses[i]: #connect the VecStim to each synapse
			# 				netcon=neuron.h.NetCon(nrn.bahl.vecstim[i]['vstim'][-1],syn.syn)
			# 				netcon.weight[0]=abs(syn.weight)
			# 				nrn.bahl.netcons[i].append(netcon)

			'event-based method'
			for n in range(spikes.shape[0]): #for each input neuron
				if spikes[n] > 0: #if this neuron spiked at this time, then
					for nrn in self.neurons: #for each bioneuron
						for syn in nrn.bahl.synapses[n]: #for each synapse conn. to input
							syn.spike_in.event(1.0*time*1000) #add a spike at time t (ms)
							# print 'event', 1.0*time*1000
							# print 'weight', syn.weight
		return step


def load_weights(P,conn_pre_label,bahl_op):
	import ipdb
	for j in range(len(bahl_op.neurons.inputs[conn_pre_label]['filenames'])): #for each bioneuron
		bahl=BahlNeuron.Bahl()
		with open(bahl_op.neurons.inputs[conn_pre_label]['filenames'][j],'r') as data_file:
			bioneuron_info=json.load(data_file)
		ens_pre_neurons=len(np.array(bioneuron_info['weights']))
		n_synapses=len(np.array(bioneuron_info['weights'])[0])
		for n in range(ens_pre_neurons):
			bahl.add_bias(bioneuron_info['bias'])
			bahl.add_connection(n)
			for i in range(n_synapses): #for each synapse connected to input neuron
				section=bahl.cell.apical(np.array(bioneuron_info['locations'])[n][i])
				weight=np.array(bioneuron_info['weights'])[n][i]
				bahl.add_synapse(n,bahl_op.P['synapse_type'],section,
										weight,P['tau'],None)
		# print bahl.bias
		bahl.start_recording()
		bahl_op.neurons.neurons[j].bahl=bahl

@Builder.register(BahlNeuron)
def build_bahlneuron(model,neuron_type,ens):
	model.sig[ens]['voltage'] = Signal(np.zeros(ens.ensemble.n_neurons),
						name='%s.voltage' %ens.ensemble.label)
	op=SimBahlNeuron(neurons=neuron_type,
						output=model.sig[ens]['out'],voltage=model.sig[ens]['voltage'],
						states=[model.time])
	model.add_op(op)

@Builder.register(nengo.Ensemble)
def build_ensemble(model,ens):
	nengo.builder.ensemble.build_ensemble(model,ens)

@Builder.register(nengo.Connection)
def build_connection(model,conn):
	import copy
	use_nrn = (
		isinstance(conn.post, nengo.Ensemble) and
		isinstance(conn.post.neuron_type, BahlNeuron))
	if use_nrn: #bioneuron connection
		rng = np.random.RandomState(model.seeds[conn])
		model.sig[conn]['in']=model.sig[conn.pre]['out']
		P=copy.copy(conn.post.neuron_type.P)
		bahl_op=conn.post.neuron_type.father_op

		#if there's no saved information about this input connection, do an optimization
		if conn.pre.label not in bahl_op.neurons.inputs:
			# if hasattr(conn.pre.neuron_type, 'inputs'):
				# P['pre_ens_pre_directory']=conn.pre.neuron_type.inputs[conn.pre.label]['directory']
			P['ens_pre_neurons']=conn.pre.n_neurons
			P['ens_pre_dim']=conn.pre.dimensions
			P['ens_pre_min_rate']=conn.pre.max_rates.low
			P['ens_pre_max_rate']=conn.pre.max_rates.high
			P['ens_pre_radius']=conn.pre.radius
			P['ens_pre_seed']=conn.pre.seed
			P['ens_pre_type']=str(conn.pre.neuron_type)
			P['ens_pre_label']=conn.pre.label
			P['ens_label']=conn.post.label
			P['ens_ideal_neurons']=conn.post.n_neurons
			P['ens_ideal_dim']=conn.post.dimensions
			P['ens_ideal_seed']=conn.post.seed
			P['ens_ideal_min_rate']=conn.post.max_rates.low
			P['ens_ideal_max_rate']=conn.post.max_rates.high
			P['ens_ideal_radius']=conn.post.radius
			from optimize_bioneuron import optimize_bioneuron
			directory=optimize_bioneuron(P)
			filenames_dir=directory+'filenames.txt'
			with open(filenames_dir,'r') as df:
				bahl_op.neurons.inputs[conn.pre.label]={'directory':directory,
																'filenames':json.load(df)}

		directory=bahl_op.neurons.inputs[conn.pre.label]['directory']
		filenames=bahl_op.neurons.inputs[conn.pre.label]['filenames']
		load_weights(P,conn.pre.label,bahl_op) 	#load weights into these operators
		conn.post.neuron_type.save_optimization(conn.pre.label,directory,filenames)	#save contents of filenames to the neuron
		model.add_op(TransmitSpikes(model.sig[conn]['in'],bahl_op,states=[model.time])) #setup spike transmission

	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)



class CustomSolver(nengo.solvers.Solver):
	import json
	import ipdb
	import copy

	def __init__(self,P,ens_pre,ens_post,inputs,A,B):
		self.P=P
		self.ens_pre=ens_pre #only used to grab the SimBahlOp operator
		self.ens_post=ens_post
		self.inputs=inputs
		self.A=np.array(A)
		self.B=np.array(B)
		self.weights=False #decoders not weights
		self.bahl_op=None
		self.activities=None
		self.upsilon=None
		self.decoders=None
		self.solver=nengo.solvers.LstsqL2()

		'''Principle 3 for Adaptive Neurons with Supervision
		Developed by Aaron Voelker
		https://github.com/arvoelke/nengolib/blob/osc-adapt/
			doc/notebooks/research/2d_oscillator_adaptation.ipynb
		'''
		if self.P['rate_decode']=='oracle':
			import nengolib
			from optimize_bioneuron import make_signal
			print 'Decoder calculation...'
			ABCD = (self.A, self.B, np.eye(2), [[0],[0]])
			# Apply discrete principle 3 to the linear system (A, B, C, D)
			msys = nengolib.synapses.ss2sim(map(np.asarray, ABCD), 
									nengo.Lowpass(self.P['tau']), dt=self.P['dt_nengo'])
			assert np.allclose(msys.C, np.eye(2))  # the remaining code assumes identity readout
			assert np.allclose(msys.D, 0)  # and no passthrough
			raw_signal=make_signal(self.P)
			# raw_signal=np.zeros_like(raw_signal)
			# raw_signal[0,:400]=0.3*np.ones(400)
			# raw_signal[1,:300]=0.5*np.ones(300)

			with nengo.Network() as decoder_model:
				u = nengo.Node(lambda t: raw_signal[:,int(t/self.P['dt_nengo'])])
				ideal_input=nengo.Node(size_in=len(msys))
				ens_in=nengo.Ensemble(n_neurons=self.P['ens_pre_neurons'],dimensions=self.P['dim'],
									seed=self.P['ens_pre_seed'],label='pre',
									radius=self.P['radius_ideal'],
									max_rates=nengo.dists.Uniform(self.P['min_ideal_rate'],
																	self.P['max_ideal_rate']))
				x_approx = nengo.Ensemble(self.ens_post.n_neurons,len(msys),
									neuron_type=BahlNeuron(self.P,self.inputs),
									# neuron_type=self.ens_post.neuron_type,
									label=self.ens_post.label,seed=self.ens_post.seed,
									radius=self.ens_post.radius,
									max_rates=self.ens_post.max_rates)
				x_direct = nengo.Ensemble(1,len(msys),neuron_type=nengo.Direct())
				ideal_output = nengo.Node(size_in=len(msys))

				nengo.Connection(u,ideal_input,synapse=None, transform=msys.B) #self.P['tau']
				nengo.Connection(ideal_input, ens_in, synapse=None)
				nengo.Connection(ens_in, x_approx, synapse=None) #does this mess up the method?
				nengo.Connection(ideal_input, x_direct, synapse=None)
				nengo.Connection(x_direct, ideal_output, synapse=None, transform=msys.A)
				nengo.Connection(ideal_output,ideal_input,synapse=self.P['tau'])

				p_u = nengo.Probe(u, synapse=None)
				p_ideal_input = nengo.Probe(ideal_input, synapse=None)
				p_ens_in=nengo.Probe(ens_in,synapse=self.P['tau'])
				p_ideal_output = nengo.Probe(ideal_output, synapse=None)
				p_approx_neurons = nengo.Probe(x_approx.neurons, 'spikes')


			with nengo.Simulator(decoder_model, dt=self.P['dt_nengo']) as decoder_sim:
				decoder_sim.run(self.P['t_train'])
			neuron.init()

			# lpf=nengo.Lowpass(self.P['kernel']['tau'])
			lpf=nengo.Lowpass(self.P['tau'])
			self.activities=lpf.filt(decoder_sim.data[p_approx_neurons],dt=self.P['dt_nengo'])
			self.upsilon=lpf.filt(decoder_sim.data[p_ideal_output],dt=self.P['dt_nengo'])
			self.decoders,self.info=self.solver(self.activities,self.upsilon)

			import matplotlib.pyplot as plt
			import seaborn as sns
			from nengo.utils.matplotlib import rasterplot
			sns.set(context='poster')
			figure1, ((ax0,a),(ax1,b),(ax2,c),(ax3,d),(ax4,e)) = plt.subplots(5,2,sharex=True)
			ax0.plot(decoder_sim.trange(),decoder_sim.data[p_u])
			ax0.set(ylabel='u')
			ax1.plot(decoder_sim.trange(),decoder_sim.data[p_ideal_input])
			ax1.set(ylabel='ideal input')
			ax2.plot(decoder_sim.trange(),decoder_sim.data[p_ens_in])
			ax2.set(ylabel='ens_in')
			rasterplot(decoder_sim.trange(),decoder_sim.data[p_approx_neurons],ax=ax3,use_eventplot=True)
			ax3.set(ylabel='bio spikes')
			ax4.plot(decoder_sim.trange(),self.activities)
			ax4.set(ylabel='bio rates')
			figure1.savefig('oracle.png')

	 	'''spike-approach:'''
 		if self.P['rate_decode']=='simulate':
			from optimize_bioneuron import make_signal
			raw_signal=make_signal(self.P)

			with nengo.Network() as decoder_model:
				stim = nengo.Node(lambda t: raw_signal[:,int(t/self.P['dt_nengo'])])
				pre=nengo.Ensemble(n_neurons=self.P['ens_pre_neurons'],dimensions=self.P['dim'],
									seed=self.P['ens_pre_seed'],label='pre',
									radius=self.P['radius_ideal'],
									max_rates=nengo.dists.Uniform(self.P['min_ideal_rate'],
																	self.P['max_ideal_rate']))
				ideal = nengo.Ensemble(n_neurons=self.ens_post.n_neurons,
									dimensions=self.P['dim'],
									neuron_type=BahlNeuron(self.P,self.inputs),
									label=self.ens_post.label,seed=self.ens_post.seed,
									radius=self.ens_post.radius,
									max_rates=self.ens_post.max_rates)

				nengo.Connection(stim,pre,synapse=None)
				nengo.Connection(pre,ideal,synapse=P['tau'])
				# nengo.Connection(pre,ideal,synapse=P['tau'],transform=P['tau'])
				# nengo.Connection(ideal,ideal,synapse=P['tau'])

				p_stim=nengo.Probe(stim,synapse=None)
				p_ideal_neurons=nengo.Probe(ideal.neurons,'spikes')

			with nengo.Simulator(decoder_model, dt=self.P['dt_nengo']) as decoder_sim:
				decoder_sim.run(self.P['t_train'])

			lpf=nengo.Lowpass(self.P['tau'])
			self.activities=lpf.filt(decoder_sim.data[p_ideal_neurons],dt=self.P['dt_nengo'])
			self.upsilon=lpf.filt(decoder_sim.data[p_stim],dt=self.P['dt_nengo'])
			self.decoders,self.info=self.solver(self.activities,self.upsilon)

	def __call__(self,A,Y,rng=None,E=None): #function that gets called by the builder
		'''
		preloaded spike approach: load activities and eval_opints from optimize_bioneuron
		'''
		if self.P['rate_decode']=='ideal':
			self.activities=self.ens_post.neuron_type.inputs[self.ens_pre.label]['ideal_rates']
			self.upsilon=self.ens_post.neuron_type.inputs[self.ens_pre.label]['signal_in']
			self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
		elif self.P['rate_decode']=='bio':
			self.activities=self.ens_post.neuron_type.inputs[self.ens_pre.label]['bio_rates']
			self.upsilon=self.ens_post.neuron_type.inputs[self.ens_pre.label]['signal_in']
			self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
		return self.decoders, dict()

		# '''1. Generate white noise signal'''
		# print 'Computing A and Y...'
		# from optimize_bioneuron import make_signal
		# #todo: pass P dictionary around
		# P=self.P
		# raw_signal=make_signal(P)
		# self.bahl_op=self.conn.post.neuron_type.father_op

		# ''' 2. Pass signal through pre LIF population, generate spikes'''
		# import nengo
		# import numpy as np
		# import pandas as pd
		# import ipdb
		# with nengo.Network() as decode_model:
		# 	signal_decode = nengo.Node(lambda t: raw_signal[:,int(t/P['dt_nengo'])]) #all dim, index at t
		# 	pre_decode = nengo.Ensemble(n_neurons=P['ens_pre_neurons'],
		# 							dimensions=P['dim'],
		# 							seed=P['ens_pre_seed'])
		# 	nengo.Connection(signal_decode,pre_decode,synapse=None)
		# 	probe_signal_decode = nengo.Probe(signal_decode)
		# 	probe_pre_decode = nengo.Probe(pre_decode.neurons,'spikes')
		# with nengo.Simulator(decode_model,dt=P['dt_nengo']) as sim_decode:
		# 	sim_decode.run(P['t_train'])
		# signal_in=sim_decode.data[probe_signal_decode]
		# spikes_in=sim_decode.data[probe_pre_decode]

		# '''3. New bioneurons, send spikes to bioneurons, collect spikes from bioneurons'''
		# from optimize_bioneuron import make_bioneuron,connect_bioneuron,run_bioneuron
		# bioneurons=[]
		# for b in range(P['n_bio']):
		# 	with open(self.bahl_op.neurons.filenames[b],'r') as data_file:
		# 		bioneuroens_pre_neuronsfo=json.load(data_file)
		# 	bias=bioneuroens_pre_neuronsfo['bias']
		# 	weights=np.zeros((P['ens_pre_neurons'],P['n_syn']))
		# 	locations=np.zeros((P['ens_pre_neurons'],P['n_syn']))
		# 	for n in range(P['ens_pre_neurons']):
		# 		for i in range(P['n_syn']):
		# 			weights[n][i]=np.array(bioneuroens_pre_neuronsfo['weights'])[n][i]
		# 			locations[n][i]=np.array(bioneuroens_pre_neuronsfo['locations'])[n][i]
		# 	bioneuron = make_bioneuron(P,weights,locations,bias)
		# 	connect_bioneuron(P,spikes_in,bioneuron)
		# 	bioneurons.append(bioneuron)
		# run_bioneuron(P)

		# '''4. Collect spikes from bioneurons'''
		# from optimize_bioneuron import get_rates
		# bio_rates=[]
		# for nrn in bioneurons:
		# 	bio_spike, bio_rate=get_rates(P,np.round(np.array(nrn.spikes),decimals=3))
		# 	bio_rates.append(bio_rate)

		# '''5. Assemble A and Y'''
		# # ipdb.set_trace()
		# self.activities=np.array(bio_rates)
		# self.upsilon=signal_in
		# self.solver=nengo.solvers.LstsqL2()
		# self.decoders,self.info=self.solver(self.activities.T,self.upsilon)

		# '''6. Reset NEURON'''
		# print 'done'
		# # neuron.init()
		# # print 'after'
		# ipdb.set_trace()
		# # self.bahl_op.neurons.neurons=None
		# # self.bahl_op.neurons.neurons=[self.bahl_op.neurons.create(i) for i in range(P['n_bio'])]
		# # # for nrn in bioneurons:
		# # 	# for n in range(P['ens_pre_neurons']):
		# # 	# 	del(nrn.synapses)
		# # 	# 	del(nrn.vecstim)
		# # 	# 	del(nrn.netcons)
		# # 	# 	nrn.synapses={}
		# # 	# 	nrn.vecstim={}
		# # 	# 	nrn.netcons={}
		# # load_weights(P,self.bahl_op)
		# # neuron.init()

		# return self.decoders, dict()