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

	def __init__(self,P,ens_pre,ens_post,model,method):
		self.P=P
		self.ens_pre=ens_pre #only used to grab the SimBahlOp operator
		self.ens_post=ens_post
		self.method=method
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
		print 'Simulating subsystem to calculate decoders for %s' %self.ens_post.label
		signals=[]
		stims=[]
		pres=[]
		synapses=[]
		transforms=[]
		connections_stim=[]
		connections_ens=[]
		probes=[]

		if self.P['rate_decode']=='bio':
			self.activities=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['bio_rates']
			self.upsilon=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['signal_in']
			self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
			return self.decoders, dict()

		with nengo.Network() as decoder_model:
			if self.method == 'simulate':
				neuron_type=BahlNeuron(self.P,father_op_inputs=self.ens_post.neuron_type.father_op.inputs)
			elif self.method == 'ideal':
				neuron_type=nengo.LIF()
			ens=nengo.Ensemble(
					n_neurons=self.ens_post.n_neurons,
					neuron_type=neuron_type,
					dimensions=self.ens_post.dimensions,
					max_rates=self.ens_post.max_rates,
					seed=self.ens_post.seed,
					radius=self.ens_post.radius,
					label=self.ens_post.label)
			p_ens_neurons=nengo.Probe(ens.neurons,'spikes')
			for n in range(len(self.input_conn)):
				signals.append(make_signal(self.P['decode']))
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
					stims[-1],pres[-1],synapse=None)) #signal not filtered before pre
				connections_ens.append(nengo.Connection(
					pres[-1],ens,synapse=synapses[-1],
					transform=transforms[-1]))
				# probes.append(nengo.Probe(stims[-1],synapse=synapses[-1]))
				probes.append(nengo.Probe(pres[-1],synapse=synapses[-1]))


		with nengo.Simulator(decoder_model, dt=self.P['dt_nengo']) as decoder_sim:
			decoder_sim.run(self.P['decode']['t_final'])

		lpf=nengo.Lowpass(self.P['tau'])
		self.activities=lpf.filt(decoder_sim.data[p_ens_neurons],dt=self.P['dt_nengo'])
		weighted_inputs=[]
		for n in range(len(self.input_conn)):
			if synapses[n]!=None:
				# weighted_inputs.append(transforms[n]*synapses[n].filt(signals[n],dt=self.P['dt_nengo']))
				weighted_inputs.append(transforms[n]*decoder_sim.data[probes[n]].T)
			else:
				weighted_inputs.append(transforms[n]*signals[n])

		self.upsilon=np.sum(np.array(weighted_inputs),axis=0).T[:len(decoder_sim.trange())]
		self.decoders,self.info=self.solver(self.activities,self.upsilon)
		import matplotlib.pyplot as plt
		import seaborn as sns
		sns.set(context='poster')
		figureA,axA=plt.subplots(1,1)
		axA.plot(decoder_sim.trange(),lpf.filt(self.upsilon,dt=self.P['dt_nengo']),label='$x(t)$')
		axA.plot(decoder_sim.trange(),np.dot(self.activities,self.decoders),label='$\hat{x}(t)$')
		axA.set(xlabel='time',ylabel='value',title='rmse=%.3f'%self.info['rmses'][0])
		axA.legend()
		figureA.savefig('decoder_accuracy.png')

		neuron.init()
		del decoder_model
		del decoder_sim
		return self.decoders, dict()

		# '''
		# # preloaded spike approach: load activities and eval_opints from optimize_bioneuron
		# # '''
		# #todo: concatenate activities and upsilon weighted by transform
		# if self.P['rate_decode']=='ideal':
		# 	self.activities=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['ideal_rates']
		# 	self.upsilon=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['signal_in']
		# 	self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
		# el
