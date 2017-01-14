import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal
import copy
import ipdb
import json
from BahlNeuron import BahlNeuron
from optimize_bioneuron import make_signal,ch_dir
import nengolib
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot

class IdentitySolver(nengo.solvers.Solver):
	def __init__(self,P):
		self.P=P
		self.weights=False #decoders not weights
		self.decoders=None
	def __call__(self,A,Y,rng=None,E=None):
		self.decoders=np.ones((P['n_bio'],P['dim']))
		return self.decoders,dict()

class CustomSolver(nengo.solvers.Solver):
	import json
	import ipdb
	import copy

	def __init__(self,P,ens_pre,ens_post,model,recurrent=False):
		self.P=P
		self.ens_pre=ens_pre #only used to grab the SimBahlOp operator
		self.ens_post=ens_post
		self.recurrent=recurrent
		self.input_conn=list(self.find_input_connections(model))
		self.weights=False #decoders not weights
		self.bahl_op=None
		self.activities=None
		self.upsilon=None
		self.decoders=None
		self.solver=nengo.solvers.LstsqL2()

	def find_input_connections(self,model):
		for c in model.all_connections:
			if c.post_obj is self.ens_post:
				yield c

	def __call__(self,A,Y,rng=None,E=None): #function that gets called by the builder

		if self.decoders != None:
			return self.decoders, dict()

		'''reconstruct all inputs, feed each with white noise, transform connection
		to ens_bio, upsilon is weighted sum of all white noise signals, activities
		is filtered spikes of ens_bio'''
		if self.P['rate_decode']=='simulate':
			print 'Simulating subsystem to calculate decoders for %s' %self.ens_post.label
			signals=[]
			stims=[]
			pres=[]
			synapses=[]
			transforms=[]
			# transforms=[self.P['my_transform'],self.P['my_transform2']]
			connections_stim=[]
			connections_bio=[]
			probes=[]



			with nengo.Network() as decoder_model:
				if isinstance(self.ens_post.neuron_type, BahlNeuron):
					neuron_type=BahlNeuron(self.P,
						father_op_inputs=self.ens_post.neuron_type.father_op.inputs)
				else:
					neuron_type=self.ens_post.neuron_type
				bio=nengo.Ensemble(
						n_neurons=self.ens_post.n_neurons,
						neuron_type=neuron_type,
						dimensions=self.ens_post.dimensions,
						max_rates=self.ens_post.max_rates,
						seed=self.ens_post.seed,
						radius=self.ens_post.radius,
						label='ens_bio')
				p_bio_neurons=nengo.Probe(bio.neurons,'spikes')
				for n in range(len(self.input_conn)):
					signals.append(make_signal(self.P,'train'))
					stims.append(nengo.Node(lambda t: signals[-1][:,int(t/self.P['dt_nengo'])]))
					pres.append(nengo.Ensemble(
						n_neurons=self.input_conn[n].pre_obj.n_neurons,
						dimensions=self.input_conn[n].pre_obj.dimensions,
						max_rates=self.input_conn[n].pre_obj.max_rates,
						seed=self.input_conn[n].pre_obj.seed,
						radius=self.input_conn[n].pre_obj.radius,
						label=self.input_conn[n].pre_obj.label))
					synapses.append(self.input_conn[n].synapse)
					transforms.append(self.input_conn[n].transform)
					connections_stim.append(nengo.Connection(
						stims[-1],pres[-1],synapse=None))
					connections_bio.append(nengo.Connection(
						pres[-1],bio,synapse=synapses[-1],
						transform=transforms[-1])) #already trained to have this transform
					# probes.append(nengo.Probe(stims[-1],synapse=synapses[-1])) #smooth with probes
					probes.append(nengo.Probe(pres[-1],synapse=synapses[-1])) #smooth with probes
				if self.recurrent:
					signals.append(make_signal(self.P,'train'))
					stims.append(nengo.Node(lambda t: signals[-1][:,int(t/self.P['dt_nengo'])]))
					#use ideal lif neurons to simulate recurrent spikes
					self_copy = nengo.Ensemble(
						n_neurons=self.ens_post.n_neurons,
						# neuron_type=neuron_type,
						dimensions=self.ens_post.dimensions,
						max_rates=self.ens_post.max_rates,
						seed=self.ens_post.seed,
						radius=self.ens_post.radius,
						label=self.ens_post.label)
					synapses.append(self.P['tau']) #todo: synapse load
					transforms.append(1.0) #todo: transform load
					connections_stim.append(nengo.Connection(stims[-1],self_copy,synapse=None))
					connections_bio.append(nengo.Connection(self_copy,bio,synapse=synapses[-1]))
					# connections_bio.append(nengo.Connection(bio,self_copy,synapse=self.P['tau']))
					probes.append(nengo.Probe(self_copy,synapse=synapses[-1]))



			with nengo.Simulator(decoder_model, dt=self.P['dt_nengo']) as decoder_sim:
				decoder_sim.run(self.P['t_train'])
			lpf=nengo.Lowpass(self.P['tau'])
			self.activities=lpf.filt(decoder_sim.data[p_bio_neurons],dt=self.P['dt_nengo'])
			weighted_inputs=[]
			for n in range(len(self.input_conn)):
				weighted_inputs.append(transforms[n]*decoder_sim.data[probes[n]])
			if self.recurrent:
				weighted_inputs.append(transforms[-1]*decoder_sim.data[probes[-1]])
			self.upsilon=np.sum(np.array(weighted_inputs),axis=0)
			self.decoders,self.info=self.solver(self.activities,self.upsilon)
			neuron.init()


		'''
		preloaded spike approach: load activities and eval_opints from optimize_bioneuron
		'''
		#todo: concatenate activities and upsilon weighted by transform
		if self.P['rate_decode']=='ideal':
			self.activities=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['ideal_rates']
			self.upsilon=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['signal_in']
			self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
		elif self.P['rate_decode']=='bio':
			self.activities=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['bio_rates']
			self.upsilon=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['signal_in']
			# self.upsilon*=self.P['my_transform']
			self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
			# import matplotlib.pyplot as plt
			# plt.plot(np.dot(self.activities.T,self.decoders))
			# plt.plot(self.upsilon)
			# ipdb.set_trace()
		return self.decoders, dict()