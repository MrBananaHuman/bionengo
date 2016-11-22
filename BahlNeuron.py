import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal

class BahlNeuron(nengo.neurons.NeuronType):
	'''compartmental neuron from Bahl et al 2012'''

	probeable=('spikes','voltage')

	def __init__(self):
		super(BahlNeuron,self).__init__()

	def create(self,dt_neuron=0.0001,dt_nengo=0.001,tau=0.01,synapse_type='ExpSyn'):
		self.tau=tau
		self.syn_type=synapse_type
		self.param_file=None
		assert dt_nengo >= dt_neuron
		self.dt_neuron=dt_neuron
		self.dt_nengo=dt_nengo
		self.delta_t=dt_nengo*1000
		self.spike_train=[]
		self.x_sample=None
		self.A_ideal=None
		self.A_actual=None
		neuron.init()

	def load_cell(self,param_file):
		import json
		from neurons import Bahl
		self.param_file=param_file
		f=open(param_file,'r')
		file=json.load(f)
		bioneuron=Bahl()
		with open(file,'r') as data_file: 
			bioneuron_info=json.load(data_file)
		for n in range(self.n_in):
			bioneuron.add_bias(bioneuron_info['bias'])
			bioneuron.add_connection(n)
			for i in range(self.n_syn):
				section=bioneuron.cell.apical(np.array(bioneuron_info['locations'])[n][i])
				weight=np.array(bioneuron_info['weights'])[n][i]
				bioneuron.add_synapse(n,self.syn_type,section,weight,self.tau,None)
		bioneuron.start_recording()
		self.x_sample=np.array(bioneuron_info['x_sample'])
		self.A_ideal=np.array(bioneuron_info['A_ideal']).T
		self.A_actual=np.array(bioneuron_info['A_actual']).T
		return bioneuron

	def rates(self, x, gain, bias): #how to get nengo to ignore rates, gain_bias?
		return np.ones(len(x))

	def gain_bias(self, max_rates, intercepts):
		return np.ones(len(max_rates)),np.ones(len(max_rates))

	def step_math(self,dt,J,output,cells,voltage): #for ALL neurons?
		#x is an array, size n_in, of whether input neurons spiked at time=t
		# for n in range(self.n_in):
		# 	if x[n] > 0:
		# 		#for all bioneurons, add a spike to all synapses connected to input neuron
		# 		for bioneuron in cells:
		# 			for syn in bioneuron.synapses[n]:
		# 				syn.spike_in.event(1.0*t*1000) #convert from s to ms
		desired_t=t*1000
		neuron.run(desired_t)
		output=[] #will this clear the inputs to step_math?
		voltage=[]
		for bioneuron in self.biopop:
			spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
			if len(spike_times) == 0:
				output.append(0)
			else:
				count=np.sum(spike_times>(t-self.dt_nengo)*1000)
				output.append(count)
				bioneuron.nengo_spike_times.extend(
						spike_times[spike_times>(t-self.dt_nengo)*1000])
			volt=np.array(bioneuron.v_record)[-1]
			bioneuron.nengo_voltages.append(volt)
			voltage.append(volt)
		output=np.array(output)/self.dt_nengo
		self.spike_train.append(output)


class SimBahlNeuron(Operator):
	def __init__(self,neurons,J,output,voltage):
		super(SimBahlNeuron,self).__init__()
		self.neurons=neurons
		self.J=J #is this input current or any type of input?
		self.output=output
		self.voltage=voltage
		self.reads = [J]
		self.sets=[output,voltage]
		self.updates=[]
		self.incs=[]
		self.cells=[self.neurons.create() for i in range(J.shape[0])]

	def make_step(self,signals,dt,rng):
		J=signals[self.J]
		output=signals[self.output]
		voltage=signals[self.voltage]
		def step_nrn():
			self.neurons.step_math(dt,J,output,self.cells,voltage)
		return step_nrn


class TransmitSpikes(Operator):
	def __init__(self,spikes,biopop):
		self.spikes=spikes
		self.biopop=biopop
		self.reads=[spikes]
		self.updates=[]
		self.sets=[]
		self.incs=[]

	def make_step(self,signals,dt,rng):
		spikes=signals[self.spikes]
		def step():
			for n in range(spikes.shape[0]): #for each input neuron
				if spikes[n] > 0:
					#for all bioneurons, add a spike to synapses connected to input neuron
					for bioneuron in self.biopop.cells:
						for syn in bioneuron.synapses[n]:
							syn.spike_in.event(1.0*t*1000) #convert from s to ms
		return step


@Builder.register(BahlNeuron)
def build_bahlneuron(model,neuron_type,ens):
	model.sig[ens]['voltage'] = Signal(np.zeros(ens.ensemble.n_neurons))
	op=SimBahlNeuron(neurons=neuron_type,J=model.sig[ens]['in'],
						output=model.sig[ens]['out'],voltage=model.sig[ens]['voltage'])
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
		model.sig[conn]['in']=model.sig[conn.pre]['out']
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
		filenames='/home/pduggins/bionengo/'+'data/QMEPDX446/'+'filenames.txt'

		if filenames == None:
			filenames=optimize_bioneuron(ens_in_seed,n_in,n_bio,n_syn,
										evals,dt_neuron,dt_nengo,tau,syn_type)
		for n in range(n_bio):
			print model.neurons[conn.post]
			conn.post.neurons[n].load_cell(filenames[n]) #grab one neuron and call a method
		model.add_op(TransmitSpikes(model.sign[conn]['in']),conn.post.ensemble)

	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)