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
	# Bioneuron=namedtuple('Bioneuron',['bioneuron','tau','syn_type',
	# 									'x_sample','A_ideal',
	# 									'gain_ideal','bias_ideal','A_actual'])
	def __init__(self):
		super(BahlNeuron,self).__init__()

	class Bahl():
		def __init__(self):
			neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
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
			self.bias_current.dur = 1e9  # TODO; limits simulation time
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
		filenames='/home/pduggins/bionengo/'+'data/44VF7AMHU/'+'filenames.txt' #todo: pass arg
		f=open(filenames,'r')
		files=json.load(f)
		for bio_idx in range(len(files)):
			with open(files[self.bio_idx],'r') as data_file: 
				bioneuron_info=json.load(data_file)
			self.A_ideal=bioneuron_info['A_ideal']
			self.gain_ideal=bioneuron_info['gain_ideal']
			self.bias_ideal=bioneuron_info['bias_ideal']
			self.A_actual=bioneuron_info['A_actual']
			self.x_sample=bioneuron_info['x_sample']
		self.bioneuron=None
		return copy.copy(self)
		# return self.Bioneuron(bioneuron=self.bioneuron,tau=self.tau,syn_type=self.syn_type,
		# 						x_sample=self.x_sample,A_ideal=self.A_ideal,
		# 						gain_ideal=self.gain_ideal,bias_ideal=self.bias_ideal,
		# 						A_actual=self.A_actual)

	def rates(self, x, gain, bias):
		"""Use LIFRate to approximate rates given bioneuron's
		ideal (optimized for) gain and bias."""
		J = self.gain_ideal * x + self.bias_ideal
		out = np.zeros_like(J)
		LIFRate_instance=nengo.LIFRate() #does this work like I think?
		LIFRate_instance.gain=self.gain_ideal
		LIFRate_instance.bias=self.bias_ideal
		nengo.LIFRate.step_math(LIFRate_instance, dt=1, J=J, output=out)
		return out

	def gain_bias(self, max_rates, intercepts): #how to remove this without bug?
		return np.ones(len(max_rates)),np.ones(len(max_rates))

	def step_math(self,dt,spiked,cells,voltage,time): #for all neurons
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
		# ipdb.set_trace()

'''
Builder #############################################################################3
'''

class SimBahlNeuron(Operator):
	def __init__(self,neurons,output,voltage,states):
		super(SimBahlNeuron,self).__init__()
		self.neurons=neurons #what is this used for?
		self.output=output
		self.voltage=voltage
		self.time=states[0]
		self.reads = [states[0]]
		self.sets=[output,voltage]
		self.updates=[]
		self.incs=[]
		self.conn_files=[]
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
				if spikes[n] > 0:
					for cell in self.cells: #for each bioneuron
						for syn in cell.bioneuron.synapses[n]: #for each synapse conn. to input
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


from optimize_bioneuron import optimize_bioneuron
@Builder.register(nengo.Connection)
def build_connection(model,conn):
	use_nrn = (
		isinstance(conn.post, nengo.Ensemble) and
		isinstance(conn.post.neuron_type, BahlNeuron))
	if use_nrn: #bioneuron connection
		rng = np.random.RandomState(model.seeds[conn])
		model.sig[conn]['in']=model.sig[conn.pre]['out'] #how to input spikes?
		#how to pass these arguments?
		n_in=conn.pre.n_neurons
		ens_in_seed=conn.pre.seed
		n_bio=conn.post.n_neurons
		n_syn=5
		evals=1000
		dt_neuron=0.0001
		dt_nengo=0.001
		tau=0.01
		synapse_type='ExpSyn'
		filenames='/home/pduggins/bionengo/'+'data/44VF7AMHU/'+'filenames.txt'

		# if filenames == None: #todo: input conn.pre, make sample neurons identical
		# 	filenames=optimize_bioneuron(ens_in_seed,n_in,n_bio,n_syn,
		# 								evals,dt_neuron,dt_nengo,tau,syn_type)

		def load_weights(filenames,sim_bahl_op):
			import json
			import copy
			f=open(filenames,'r')
			sim_bahl_op.conn_files=json.load(f)
			for bio_idx in range(n_bio): #for each bioneuron
				bioneuron=BahlNeuron.Bahl() #todo: less deep
				with open(sim_bahl_op.conn_files[bio_idx],'r') as data_file: 
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
				bioneuron.start_recording() #NEURON method
				sim_bahl_op.cells[bio_idx].bioneuron=bioneuron

		sim_bahl_op=model.operators[7] 			#how to grab model SimBahlNeuron operators? 
		load_weights(filenames,sim_bahl_op) 	#load weights into these operators
		model.add_op(TransmitSpikes(model.sig[conn]['in'],sim_bahl_op,states=[model.time]))
		# from nengo.builder.connection import build_linear_system
		# eval_points, activities, targets = build_linear_system(
		#     model, conn, rng=rng)
		# # account for transform
		# from nengo.utils.builder import full_transform
		# transform = full_transform(conn)
		# targets = np.dot(targets, transform.T)
		# weights, solver_info = conn.solver(
		# 	activities, targets)
			# rng=rng, E=model.params[conn.post].scaled_encoders.T)
	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)