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

class CustomSolver(nengo.solvers.Solver):
	import json
	import ipdb
	import copy

	def __init__(self,P,ens_pre,ens_post):
		self.P=P
		self.ens_pre=ens_pre #only used to grab the SimBahlOp operator
		self.ens_post=ens_post
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
									neuron_type=BahlNeuron(self.P),
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

			lpf=nengo.Lowpass(self.P['kernel']['tau'])
			self.activities=lpf.filt(decoder_sim.data[p_approx_neurons],dt=self.P['dt_nengo'])
			self.upsilon=lpf.filt(decoder_sim.data[p_ideal_output],dt=self.P['dt_nengo'])
			self.decoders,self.info=self.solver(self.activities,self.upsilon)


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
			figure1.savefig('decoder_calc_oracle.png')

	 	'''spike-approach:
	 	 1. get decoders for recurrent connection based on 
	 	activities and eval points in the feedforward case;
	 	2. use those decoders in the recurrent connection
	 	3. gather activities, set eval points as the decoded input values
	 	plus the decoded recurrent values'''
 		if self.P['rate_decode']=='simulate':
			raw_signal=make_signal(self.P)
			with nengo.Network() as decoder_model:
				stim = nengo.Node(lambda t: raw_signal[:,int(t/self.P['dt_nengo'])])
				pre=nengo.Ensemble(n_neurons=self.P['ens_pre_neurons'],dimensions=self.P['dim'],
									seed=self.P['ens_pre_seed'],label='pre',
									radius=self.P['radius_ideal'],
									max_rates=nengo.dists.Uniform(self.P['min_ideal_rate'],
																	self.P['max_ideal_rate']))
				bio = nengo.Ensemble(n_neurons=self.ens_post.n_neurons,
									neuron_type=BahlNeuron(self.P,self),
									# neuron_type=nengo.LIF(),
									dimensions=self.P['dim'],
									label=self.ens_post.label,seed=self.ens_post.seed,
									radius=self.ens_post.radius,
									max_rates=self.ens_post.max_rates)
				nengo.Connection(stim,pre,synapse=None)
				nengo.Connection(pre,bio,synapse=P['tau'])
				p_stim=nengo.Probe(stim,synapse=None)
				p_pre=nengo.Probe(pre,synapse=P['tau'])
				p_bio=nengo.Probe(bio,synapse=P['tau'])
				p_bio_neurons=nengo.Probe(bio.neurons,'spikes')
			with nengo.Simulator(decoder_model, dt=self.P['dt_nengo']) as decoder_sim:
				decoder_sim.run(self.P['t_train'])
			lpf=nengo.Lowpass(self.P['tau'])
			self.activities=lpf.filt(decoder_sim.data[p_bio_neurons],dt=self.P['dt_nengo'])
			self.upsilon=lpf.filt(decoder_sim.data[p_stim],dt=self.P['dt_nengo'])
			self.decoders,self.info=self.solver(self.activities,self.upsilon)
			# self.decoders*=(1-P['tau']) #but whyyyyy?

	def __call__(self,A,Y,rng=None,E=None): #function that gets called by the builder
		'''
		preloaded spike approach: load activities and eval_opints from optimize_bioneuron
		'''
		if self.P['rate_decode']=='ideal':
			self.activities=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['ideal_rates']
			self.upsilon=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['signal_in']
			self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
		elif self.P['rate_decode']=='bio':
			self.activities=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['bio_rates']
			self.upsilon=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['signal_in']
			self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
		return self.decoders, dict()