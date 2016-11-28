import nengo
import neuron
import numpy as np
import ipdb

class BioneuronNode(nengo.Node):
	
	def __init__(self,n_in,n_bio,n_syn,
					dt_neuron=0.0001,dt_nengo=0.001,
					tau=0.01,synapse_type='ExpSyn'):
		super(BioneuronNode,self).__init__(self.step,size_in=n_in,size_out=n_bio)
		self.n_in=n_in #? how do deal with multiple input sizes
		self.n_bio=n_bio
		self.n_syn=n_syn
		self.tau=tau
		self.syn_type=synapse_type
		assert dt_nengo >= dt_neuron
		self.dt_neuron=dt_neuron
		self.dt_nengo=dt_nengo
		self.delta_t=dt_nengo*1000
		self.spike_train=[]
		self.ens_in_seed=None
		self.evals=None
		self.filenames=None
		self.biopop=None
		self.x_sample=None
		self.A_ideal=None
		self.A_actual=None
		self.d=None
		self.decoded=[]
		neuron.h.dt = dt_neuron*1000

	def connect_to(self,ens_in_seed,evals=1000,filenames=None):
		self.ens_in_seed=ens_in_seed #make a list for multiple inputs
		self.evals=evals
		self.filenames=filenames
		if self.filenames == None:
			self.filenames=self.optimize_biopop()
		self.biopop=self.load_biopop()
		self.save_sample_activities()
		self.d=self.decoders_from_sample_activities()
		neuron.init()

	def load_biopop(self):
		import json
		from neurons import Bahl
		f=open(self.filenames,'r')
		files=json.load(f)
		biopop=[]
		for bio_idx in range(self.n_bio):
			bioneuron=Bahl()
			with open(files[bio_idx],'r') as data_file: 
				bioneuron_info=json.load(data_file)
			for n in range(self.n_in):
				bioneuron.add_bias(bioneuron_info['bias'])
				bioneuron.add_connection(n)
				for i in range(self.n_syn):
					section=bioneuron.cell.apical(np.array(bioneuron_info['locations'])[n][i])
					weight=np.array(bioneuron_info['weights'])[n][i]
					bioneuron.add_synapse(n,self.syn_type,section,weight,self.tau,None)
			bioneuron.start_recording()
			biopop.append(bioneuron)
		return biopop

	def optimize_biopop(self):
		from optimize_bioneuron import optimize_bioneuron
		filenames=optimize_bioneuron(self.ens_in_seed,self.n_in,self.n_bio,self.n_syn,
									self.evals,self.dt_neuron,self.dt_nengo,
									self.tau,self.syn_type)
		return filenames

	def save_sample_activities(self):
		import json
		f=open(self.filenames,'r')
		files=json.load(f)
		A_ideal=[]
		A_actual=[]
		x_sample=0
		for bio_idx in range(self.n_bio):
			with open(files[bio_idx],'r') as data_file: 
				bioneuron_info=json.load(data_file)
			A_ideal.append(bioneuron_info['A_ideal'])
			A_actual.append(bioneuron_info['A_actual'])
			x_sample=bioneuron_info['x_sample']
		self.x_sample=np.array(x_sample)
		self.A_ideal=np.array(A_ideal).T
		self.A_actual=np.array(A_actual).T

	def decoders_from_sample_activities(self,function=lambda x: x):
		A=np.matrix(self.A_actual)
		A_T=np.matrix(self.A_actual.T)
		f_X=np.matrix(function(self.x_sample)).T
		solver=nengo.solvers.LstsqL2()
		d,info=solver(np.array(A),np.array(f_X))
		return d

	def step(self,t,x):
		# ipdb.set_trace()
		#x is an array, size n_in, of whether input neurons spiked at time=t
		for n in range(self.n_in):
			if x[n] > 0:
				#for all bioneurons, add a spike to all synapses connected to input neuron
				for bioneuron in self.biopop:
					for syn in bioneuron.synapses[n]:
						syn.spike_in.event(1.0*t*1000) #convert from s to ms
		desired_t=t*1000
		neuron.run(desired_t) #no floating point addition errors!
		output=[]
		for bioneuron in self.biopop:
			spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
			if len(spike_times) == 0:
				output.append(0)
			else:
				count=np.sum(spike_times>(t-self.dt_nengo)*1000)
				output.append(count)
				#store spike time according to dt_nengo
				bioneuron.nengo_spike_times.extend(
						spike_times[spike_times>(t-self.dt_nengo)*1000])
			#store voltage at the end of the delta_t timestep
			bioneuron.nengo_voltages.append(np.array(bioneuron.v_record)[-1])
		output=np.array(output)/self.dt_nengo
		self.spike_train.append(output)
		# return np.dot(output,self.d)
		return output