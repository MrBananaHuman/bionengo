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
		#load attributes of optimized neuron
		f=open(self.filenames,'r')
		self.my_file=json.load(f)[self.bio_idx]
		with open(self.my_file,'r') as data_file: 
			bioneuron_info=json.load(data_file)
		self.A_ideal=bioneuron_info['A_ideal']
		self.gain_ideal=bioneuron_info['gain_ideal']
		self.bias_ideal=bioneuron_info['bias_ideal']
		self.A_actual=bioneuron_info['A_actual']
		self.x_sample=bioneuron_info['x_sample']
		self.bioneuron=None
		return copy.copy(self)

	def rates(self, x, gain, bias):
		"""Use LIFRate to approximate rates given bioneuron's
		ideal (optimized for) gain and bias."""
		#only grabs gain and bias from ONE instance of a bioneuron, so fails
			# - how to pass SimBahlNeuron.cells?
		J = self.gain_ideal * x + self.bias_ideal
		out = np.zeros_like(J)
		LIFRate_instance=nengo.LIFRate() #does this work like I think?
		LIFRate_instance.gain=self.gain_ideal
		LIFRate_instance.bias=self.bias_ideal
		nengo.LIFRate.step_math(LIFRate_instance, dt=1, J=J, output=out)
		return out

	def gain_bias(self, max_rates, intercepts): #how to remove this without error?
		#only grabs gain and bias from ONE instance of a bioneuron, so fails
			# - how to pass SimBahlNeuron.cells?
		return np.ones(len(max_rates)),np.ones(len(max_rates))

	def step_math(self,dt,spiked,cells,voltage,time):
		desired_t=time*1000
		neuron.run(desired_t)
		new_spiked=[]
		new_voltage=[]
		for cell in cells:
			spike_times=np.round(np.array(cell.bioneuron.spikes),decimals=3)
			count=np.sum(spike_times>(time-dt)*1000)
			new_spiked.append(count)
			cell.bioneuron.nengo_spike_times.extend(
					spike_times[spike_times>(time-dt)*1000])
			volt=np.array(cell.bioneuron.v_record)[-1]
			cell.bioneuron.nengo_voltages.append(volt)
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
		self.cells=[self.neurons.create(i) for i in range(output.shape[0])]
		self.dt_neuron=0.0001*1000 #todo - pass as argument
		neuron.h.dt = self.dt_neuron
		neuron.init()

	def make_step(self,signals,dt,rng):
		output=signals[self.output]
		voltage=signals[self.voltage]
		time=signals[self.time]
		def step_nrn():
			#one bahlneuron object runs step_math, but arg is all cells - what/where returns?
			self.neurons.step_math(dt,output,self.cells,voltage,time)
		return step_nrn


class TransmitSpikes(Operator):
	def __init__(self,spikes,sim_bahl_op,states):
		self.spikes=spikes
		self.cells=sim_bahl_op.cells
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
					for cell in self.cells: #for each bioneuron
						for syn in cell.bioneuron.synapses[n]: #for each synapse conn. to input
							syn.spike_in.event(1.0*time*1000) #add a spike at time t (ms)
		return step


# from weakref import WeakKeyDictionary
# ens_to_cells = WeakKeyDictionary()
@Builder.register(BahlNeuron)
def build_bahlneuron(model,neuron_type,ens):
	model.sig[ens]['voltage'] = Signal(np.zeros(ens.ensemble.n_neurons),
						name='%s.voltage' %ens.ensemble.label)
	op=SimBahlNeuron(neurons=neuron_type,
						output=model.sig[ens]['out'],voltage=model.sig[ens]['voltage'],
						states=[model.time])
	# ens_to_cells[ens.ensemble] = op.cells
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
		evals=10
		dt_neuron=0.0001
		dt_nengo=0.001
		tau=0.01
		synapse_type='ExpSyn'
		sim_bahl_op=model.operators[7] 			#how to grab model SimBahlNeuron operators? 

		if sim_bahl_op.cells[0].my_file == None: #todo: input conn.pre, make sample neurons identical
			from optimize_bioneuron import optimize_bioneuron
			ipdb.set_trace()
			filenames=optimize_bioneuron(ens_in_seed,n_in,n_bio,n_syn,
										evals,dt_neuron,dt_nengo,tau,syn_type)

		def load_weights(sim_bahl_op):
			import copy
			for j in range(n_bio): #for each bioneuron
				bioneuron=BahlNeuron.Bahl() #todo: less deep
				with open(sim_bahl_op.cells[j].my_file,'r') as data_file:
					bioneuron_info=json.load(data_file)
				n_inputs=len(np.array(bioneuron_info['weights']))
				n_synapses=len(np.array(bioneuron_info['weights'])[0])
				for n in range(n_inputs):
					bioneuron.add_bias(bioneuron_info['bias'])
					bioneuron.add_connection(n)
					for i in range(n_synapses): #for each synapse connected to input neuron
						section=bioneuron.cell.apical(np.array(bioneuron_info['locations'])[n][i])
						weight=np.array(bioneuron_info['weights'])[n][i]
						bioneuron.add_synapse(n,sim_bahl_op.cells[0].syn_type,section,
												weight,sim_bahl_op.cells[0].tau,None)
				bioneuron.start_recording()
				sim_bahl_op.cells[j].bioneuron=bioneuron

		load_weights(sim_bahl_op) 	#load weights into these operators
		model.add_op(TransmitSpikes(model.sig[conn]['in'],sim_bahl_op,states=[model.time]))

	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)

class CustomSolver(nengo.solvers.Solver):
	import json
	def __init__(self,filenames):
		self.filenames=filenames
		self.weights=False #decoders not weights
		if self.filenames==None: #do the same optimization as in build_connection()
			#self.filenames='''call optimize bioneuron '''
		#grab eval points and activities from optimization
		f=open(self.filenames,'r')
		files=json.load(f)
		self.A_ideal=[]
		self.A_actual=[]
		self.gain_ideal=[]
		self.bias_ideal=[]
		self.x_sample=[]
		for bio_idx in range(len(files)):
			with open(files[bio_idx],'r') as data_file: 
				bioneuron_info=json.load(data_file)
			self.A_ideal.append(bioneuron_info['A_ideal'])
			self.A_actual.append(bioneuron_info['A_actual'])
			self.gain_ideal.append(bioneuron_info['gain_ideal'])
			self.bias_ideal.append(bioneuron_info['bias_ideal'])
			self.x_sample.append(bioneuron_info['x_sample'])
		
		self.solver=nengo.solvers.LstsqL2()
		self.decoders,self.info=self.solver(
									np.array(self.A_actual).T,
									np.array(self.x_sample)[0])
		self.decoders=(np.ones((1,len(files)))*self.decoders).T
	def __call__(self,A,Y,rng=None,E=None): #function that gets called by the builder
		return self.decoders, dict()