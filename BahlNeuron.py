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
	def __init__(self,filenames=None):
		super(BahlNeuron,self).__init__()
		self.filenames=filenames

	class Bahl():
		def __init__(self):
			# neuron.h.load_file('/home/psipeter/bionengo/bahl.hoc') #todo
			neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc') #todo
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
			self.vecstim[idx]={'vstim':[],'vtimes':[]} #list of input spike times from this neuron
			self.netcons[idx]=[] #list of netcon objects between input vecstim and synapses for this nrn

		def add_synapse(self,idx,syn_type,section,weight,tau,tau2=0.005):
			from synapses import ExpSyn, Exp2Syn
			if syn_type == 'ExpSyn':
				self.synapses[idx].append(ExpSyn(section,weight,tau))
			elif syn_type == 'Exp2Syn':
				self.synapses[idx].append(Exp2Syn(section,weight,tau,tau2))

	def create(self,idx,tau=0.01,synapse_type='ExpSyn'):
		self.bio_idx=idx
		self.tau=tau
		self.syn_type=synapse_type
		self.bahl=None
		return copy.copy(self)

	def rates(self, x, gain, bias):
		import ipdb
		ipdb.set_trace()
		'''1. Generate white noise signal'''
		from optimize_bioneuron import make_signal
		#todo: pass P dictionary around
		P={
			'n_in':50,
			'n_syn':5,
			'n_bio':10,
			'dim':2,
			'ens_in_seed':333,
			'dt':0.0001,
			'synapse_tau':0.01,
			'synapse_type':'ExpSyn',
			't_sample':3.0,
			'min_ideal_rate':40,
			'max_ideal_rate':60,
			'n_seg': 5,
			'dx':0.01,
			'n_processes':10,
			'signal':
				{'type':'equalpower','max_freq':5.0,'mean':0.0,'std':1.0},
			'kernel': #for smoothing spikes to calculate rate for tuning curve
				#{'type':'exp','tau':0.02},
				{'type':'gauss','sigma':0.01,},
			}
		raw_signal=make_signal(P)

		ipdb.set_trace()
		''' 2. Pass signal through pre LIF population, generate spikes'''
		import nengo
		import numpy as np
		import pandas as pd
		import ipdb
		with nengo.Network() as opt_model:
			signal = nengo.Node(lambda t: raw_signal[:,int(t/P['dt'])]) #all dim, index at t
			pre = nengo.Ensemble(n_neurons=P['n_in'],
									dimensions=P['dim'],
									seed=P['ens_in_seed'])
			nengo.Connection(signal,pre,synapse=None)
			probe_signal = nengo.Probe(signal)
			probe_pre = nengo.Probe(pre.neurons,'spikes')
		with nengo.Simulator(opt_model,dt=P['dt']) as opt_sim:
			opt_sim.run(P['t_sample'])
		signal_in=opt_sim.data[probe_signal]
		spikes_in=opt_sim.data[probe_pre]

		ipdb.set_trace()
		'''3. Send to bioneurons, collect spikes from bioneurons'''
		bioneurons=[self.neurons[b].bahl for b in range(len(self.neurons))]
		for nrn in bioneurons:
			for n in range(P['n_in']):
				#create spike time vectors and an artificial spiking cell that delivers them
				vstim=neuron.h.VecStim()
				nrn.vecstim[n]['vstim'].append(vstim)
				spike_times_ms=list(1000*P['dt']*np.nonzero(spikes_in[:,n])[0]) #timely
				vtimes=neuron.h.Vector(spike_times_ms)
				nrn.vecstim[n]['vtimes'].append(vtimes)
				nrn.vecstim[n]['vstim'][-1].play(nrn.vecstim[n]['vtimes'][-1])
				#connect the VecStim to each synapse
				for syn in nrn.synapses[n]:
					netcon=neuron.h.NetCon(nrn.vecstim[n]['vstim'][-1],syn.syn)
					netcon.weight[0]=abs(syn.weight)
					nrn.netcons[n].append(netcon)
		neuron.h.dt = P['dt']*1000
		neuron.init()
		neuron.run(P['t_sample']*1000)

		ipdb.set_trace()
		'''4. Collect spikes from bioneurons'''
		from optimize_bioneuron import get_rates
		bio_rates=[]
		for nrn in bioneurons:
			bio_spike, bio_rate=get_rates(P,np.round(np.array(nrn.spikes),decimals=3))
			bio_rates.append(bio_rate)

		#5. Assemble A and Y
		ipdb.set_trace()
		out=None
		return out

	def gain_bias(self, max_rates, intercepts): #todo: remove this without errors
		return np.ones(len(max_rates)),np.ones(len(max_rates))

	def step_math(self,dt,spiked,neurons,voltage,time):
		desired_t=time*1000
		neuron.run(desired_t)
		new_spiked=[]
		new_voltage=[]
		for nrn in neurons:
			spike_times=np.round(np.array(nrn.bahl.spikes),decimals=3)
			count=np.sum(spike_times>(time-dt)*1000)
			new_spiked.append(count)
			nrn.bahl.nengo_spike_times.extend(
					spike_times[spike_times>(time-dt)*1000])
			volt=np.array(nrn.bahl.v_record)[-1]
			nrn.bahl.nengo_voltages.append(volt)
			new_voltage.append(volt)
		spiked[:]=np.array(new_spiked)/dt
		voltage[:]=np.array(new_voltage)

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
		self.dt_neuron=0.0001*1000 #todo - pass as argument
		neuron.h.dt = self.dt_neuron
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
	def __init__(self,spikes,sim_bahl_op,states):
		self.spikes=spikes
		self.neurons=sim_bahl_op.neurons.neurons
		self.time=states[0]
		self.reads=[spikes,states[0]]
		self.updates=[]
		self.sets=[]
		self.incs=[]
		neuron.init()

	def make_step(self,signals,dt,rng):
		spikes=signals[self.spikes]
		time=signals[self.time]
		def step():
			for n in range(spikes.shape[0]): #for each input neuron
				if spikes[n] > 0: #if this neuron spiked at this time, then
					for nrn in self.neurons: #for each bioneuron
						for syn in nrn.bioneuron.synapses[n]: #for each synapse conn. to input
							syn.spike_in.event(1.0*time*1000) #add a spike at time t (ms)
		return step


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
	use_nrn = (
		isinstance(conn.post, nengo.Ensemble) and
		isinstance(conn.post.neuron_type, BahlNeuron))
	if use_nrn: #bioneuron connection
		rng = np.random.RandomState(model.seeds[conn])
		model.sig[conn]['in']=model.sig[conn.pre]['out']
		#how to pass these arguments?
		n_in=conn.pre.n_neurons
		ens_in_seed=conn.pre.seed
		n_bio=conn.post.n_neurons
		n_syn=5
		dim=3
		evals=2
		dt_neuron=0.0001
		dt_nengo=0.001
		tau=0.01
		syn_type='ExpSyn'
		sim_bahl_op=model.operators[7] 			#how to grab model SimBahlNeuron operators? 

		if sim_bahl_op.neurons.filenames == None: 
			from optimize_bioneuron import optimize_bioneuron
			#todo: input conn.pre, make sample neurons identical
			filenames=optimize_bioneuron(ens_in_seed,n_in,n_bio,n_syn,dim,
											evals,dt_neuron,dt_nengo,tau,syn_type)
			with open(filenames,'r') as df:
				sim_bahl_op.neurons.filenames=json.load(df)

		def load_weights(sim_bahl_op):
			import copy
			for j in range(n_bio): #for each bioneuron
				bahl=BahlNeuron.Bahl() #todo: less deep
				# ipdb.set_trace()
				with open(sim_bahl_op.neurons.filenames[j],'r') as data_file:
					bioneuron_info=json.load(data_file)
				n_inputs=len(np.array(bioneuron_info['weights']))
				n_synapses=len(np.array(bioneuron_info['weights'])[0])
				for n in range(n_inputs):
					bahl.add_bias(bioneuron_info['bias'])
					bahl.add_connection(n)
					for i in range(n_synapses): #for each synapse connected to input neuron
						section=bahl.cell.apical(np.array(bioneuron_info['locations'])[n][i])
						weight=np.array(bioneuron_info['weights'])[n][i]
						bahl.add_synapse(n,sim_bahl_op.neurons.neurons[0].syn_type,section,
												weight,sim_bahl_op.neurons.neurons[0].tau,None)
				bahl.start_recording()
				sim_bahl_op.neurons.neurons[j].bahl=bahl

		load_weights(sim_bahl_op) 	#load weights into these operators
		model.add_op(TransmitSpikes(model.sig[conn]['in'],sim_bahl_op,states=[model.time]))

	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)

# class CustomSolver(nengo.solvers.Solver):
# 	import json
# 	def __init__(self,filenames):
# 		self.filenames=filenames
# 		self.weights=False #decoders not weights
# 		#spike-approach: feed in N-dim white noise signal, then construct multidimensional A and Y


# 		#response-curve approach - doesn't work for dim>1
# 		if self.filenames==None: #do the same optimization as in build_connection()
# 			print self.filenames
# 			#self.filenames='''call optimize bioneuron '''
# 		#grab eval points and activities from optimization
# 		f=open(self.filenames,'r')
# 		files=json.load(f)
# 		self.A_ideal=[]
# 		self.A_actual=[]
# 		self.gain_ideal=[]
# 		self.bias_ideal=[]
# 		self.encoders_ideal=[]
# 		self.x_sample=[]
# 		for bio_idx in range(len(files)):
# 			with open(files[bio_idx],'r') as data_file: 
# 				bioneuron_info=json.load(data_file)
# 			self.A_ideal.append(np.asfarray(bioneuron_info['A_ideal']).tolist())
# 			self.A_actual.append(np.asfarray(bioneuron_info['A_actual']).tolist())
# 			self.gain_ideal.append(np.asfarray(bioneuron_info['gain_ideal']).tolist())
# 			self.bias_ideal.append(np.asfarray(bioneuron_info['bias_ideal']).tolist())
# 			self.encoders_ideal.append(np.asfarray(bioneuron_info['encoders_ideal']).tolist())
# 			self.x_sample.append(np.asfarray(bioneuron_info['x_sample']).tolist())
# 		self.A_ideal=np.array(self.A_ideal)
# 		self.A_actual=np.array(self.A_actual)
# 		self.gain_ideal=np.array(self.gain_ideal)
# 		self.bias_ideal=np.array(self.bias_ideal)
# 		self.encoders_ideal=np.array(self.encoders_ideal)
# 		self.x_sample=np.array(self.x_sample)
# 		self.solver=nengo.solvers.LstsqL2()
# 		self.A=self.A_actual.T
# 		#todo - is this at all valid?
# 		self.Y=np.array([self.x_sample[0] for dim in range(self.encoders_ideal.shape[1])]).T
# 		self.decoders,self.info=self.solver(A,Y)
# 		# self.decoders=(np.ones((1,len(files)))*self.decoders).T
# 	def __call__(self,A,Y,rng=None,E=None): #function that gets called by the builder
# 		return self.decoders, dict()