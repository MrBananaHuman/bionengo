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

	def __init__(self,P,ens_post,model,method):
		self.P=P
		self.ens_post=ens_post
		self.method=method
		 #todo: creating a full input_conn requires creating solver in correct place in test
		self.input_conn=list(self.find_input_connections(model,self.ens_post))
		self.bioensembles=list(self.find_bioensembles(model))
		self.weights=False #decoders not weights
		self.bahl_op=None
		self.activities=None
		self.upsilon=None
		self.decoders=None
		self.solver=nengo.solvers.LstsqL2()

	def find_input_connections(self,model,ens_post):
		for c in model.all_connections:
			if c.post_obj is ens_post:
				yield c

	def find_bioensembles(self,model):
		for c in model.all_connections:
			if isinstance(c.post_obj.neuron_type, BahlNeuron):
				yield c.post_obj

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
		connections_ideal=[]
		probes=[]
		probes_stim=[]
		probes_spikes=[]

		# ipdb.set_trace()
		if self.P['decoder_train']=='load_bio' and self.method != 'ideal': #no ens_pre anymore
			self.activities=self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['bio_rates']
			self.upsilon=nengo.Lowpass(self.P['kernel']['tau']).filt(
				self.ens_post.neuron_type.father_op.inputs[self.ens_pre.label]['signal_in'],
				dt=self.P['dt_nengo'])
			self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
			return self.decoders, dict()

		with nengo.Network(label='decoder_model') as decoder_model:
			if self.method == 'bio':
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
			ideal=nengo.Ensemble(
					n_neurons=self.ens_post.n_neurons,
					neuron_type=nengo.LIF(),
					dimensions=self.ens_post.dimensions,
					max_rates=self.ens_post.max_rates,
					seed=self.ens_post.seed,
					radius=self.ens_post.radius)
			p_ens_neurons=nengo.Probe(ens.neurons,'spikes')
			p_ideal_neurons=nengo.Probe(ideal.neurons,'spikes')

			for n in range(len(self.input_conn)):
				signals.append(make_signal(self.P['decode']))
				stims.append(nengo.Node(lambda t, n=n: signals[n][:,np.floor(t/self.P['dt_nengo'])]))
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
					stims[n],pres[n],synapse=None)) #signal not filtered before pre
				connections_ens.append(nengo.Connection(
					pres[n],ens,synapse=synapses[n],
					transform=transforms[n]))
				connections_ideal.append(nengo.Connection(
					pres[n],ideal,synapse=synapses[n],
					transform=transforms[n]))
				probes_stim.append(nengo.Probe(stims[n],synapse=None))
				probes.append(nengo.Probe(pres[n],synapse=synapses[n]))
				probes_spikes.append(nengo.Probe(pres[n].neurons,'spikes'))


		with nengo.Simulator(decoder_model, dt=self.P['dt_nengo']) as decoder_sim:
			decoder_sim.run(self.P['decode']['t_final'])

		lpf=nengo.Lowpass(self.P['kernel']['tau'])
		self.activities_nengo=lpf.filt(decoder_sim.data[p_ens_neurons],dt=self.P['dt_nengo'])

		weighted_inputs=[]
		for n in range(len(self.input_conn)):
			if synapses[n]!=None:
				filt1=synapses[n].filt(transforms[n]*signals[n][0],dt=self.P['dt_nengo'])
				filt2=synapses[n].filt(filt1,dt=self.P['dt_nengo'])
				weighted_inputs.append(filt2)
			else:
				weighted_inputs.append(transforms[n]*signals[n][0])

		self.upsilon=np.sum(np.array(weighted_inputs),axis=0).T[:len(decoder_sim.trange())]
		self.decoders,self.info=self.solver(self.activities_nengo,self.upsilon)
		if len(self.decoders.shape)==1:
			self.decoders=self.decoders.reshape((self.decoders.shape[0],1))

		# if self.method == 'bio' and self.ens_post.label=='ens_bio2':
		# 	pre_spikes_count=0
		# 	pre_spikes_nonzero=[]
		# 	for n in range(len(probes_spikes)):
		# 		pre_spikes_count+=int(np.sum(decoder_sim.data[probes_spikes[n]])*self.P['dt_nengo'])
		# 		spike_times=[]
		# 		for m in range(decoder_sim.data[probes_spikes[n]].shape[1]):
		# 			spike_times.append(np.nonzero(decoder_sim.data[probes_spikes[n]][:,m])[0])
		# 		pre_spikes_nonzero.append(spike_times)
		# 	bio_spikes_nengo=decoder_sim.data[p_ens_neurons]
		# 	bio_spikes_NEURON=np.array([np.array(nrn.spikes) for nrn in ens.neuron_type.neurons])
		# 	bio_spikes_train=ens.neuron_type.father_op.inputs[self.ens_pre.label]['bio_spikes']
		# 	print 'decode %s pre spikes:'%ens.label, pre_spikes_count
		# 	print pre_spikes_nonzero
		# 	for n in range(self.P['n_bio']):
		# 		print 'decode %s neuron %s nengo spikes:'%(ens.label,n), int(np.sum(bio_spikes_nengo[:,n])*self.P['dt_nengo'])
		# 		print np.nonzero(bio_spikes_nengo[:,n])
				# print 'decode %s neuron %s NEURON spikes:'%(ens.label,n), len(bio_spikes_NEURON[n])
				# print bio_spikes_NEURON[n]
				# print 'decode %s neuron %s weight [0,0]:'%(ens.label,n)
				# print ens.neuron_type.father_op.inputs[self.ens_pre.label]['weights'][n][0,0]
				# print 'decode %s neuron %s training spikes:'%(ens.label,n), int(np.sum(bio_spikes_train[n])*self.P['dt_nengo'])
				# print np.nonzero(bio_spikes_train[n])
				# print 'decode %s neuron %s NEURON voltage:'%(ens.label,n)
				# print np.array(ens.neuron_type.neurons[n].v_record)

		import matplotlib.pyplot as plt
		import seaborn as sns
		sns.set(context='poster')
		figureA,axA=plt.subplots(1,1)
		axA.plot(decoder_sim.trange(),self.upsilon,label='$x(t)$')
		axA.plot(decoder_sim.trange(),np.dot(self.activities_nengo,self.decoders),label='$\hat{x}(t)$')
		axA.set(xlabel='time',ylabel='value',title='rmse=%.3f'%self.info['rmses'])
		axA.legend()
		figureA.savefig('decoder_accuracy_%s.png'%self.ens_post.label)

		figureB, (axA,axB) = plt.subplots(1,2,sharex=True)
		rasterplot(decoder_sim.trange(),decoder_sim.data[p_ens_neurons],ax=axA,use_eventplot=True)
		axA.set(ylabel='bio spikes',yticks=([]),title='decode')
		rasterplot(decoder_sim.trange(),decoder_sim.data[p_ideal_neurons],ax=axB,use_eventplot=True)
		axB.set(ylabel='ideal spikes',yticks=([]))
		figureB.savefig('decoder_spikes_%s.png'%self.ens_post.label)

		if self.method == 'bio': #reset neuron state of bahl neurons in test simulation
			# ipdb.set_trace()
			if self.ens_post.label=='ens_bio':
				figureC,(ax1,ax2,ax3)=plt.subplots(3,1,sharex=True)
				rasterplot(decoder_sim.trange(),decoder_sim.data[probes_spikes[0]],ax=ax1,use_eventplot=True)
				ax1.set(ylabel='pre spikes',yticks=([]))			
				# rasterplot(decoder_sim.trange(),decoder_sim.data[probes_spikes[1]],ax=ax2,use_eventplot=True)
				# ax2.set(ylabel='pre2 spikes',yticks=([]))			
				# rasterplot(decoder_sim.trange(),decoder_sim.data[probes_spikes[2]],ax=ax3,use_eventplot=True)
				# ax3.set(ylabel='pre3 spikes',yticks=([]))
				figureC.savefig('pre_spikes_%s.png'%self.ens_post.label)
			if self.ens_post.label=='ens_bio2':
				figureD,ax1=plt.subplots(1,1,sharex=True)
				rasterplot(decoder_sim.trange(),decoder_sim.data[probes_spikes[0]],ax=ax1,use_eventplot=True)
				ax1.set(ylabel='pre spikes',yticks=([]))			
				figureD.savefig('pre_spikes_%s.png'%self.ens_post.label)
				# axV.plot(np.array(ens.neuron_type.neurons[0].t_record),np.array(ens.neuron_type.neurons[0].v_record))
			from synapses import ExpSyn
			for ens in self.bioensembles:
				for nrn in ens.neuron_type.neurons:
					nrn.add_cell()
				for inpt in ens.neuron_type.father_op.inputs.iterkeys():
					for nrn in range(len(ens.neuron_type.neurons)):
						bioneuron=ens.neuron_type.neurons[nrn]
						# bioneuron.add_cell()
						with open(ens.neuron_type.father_op.inputs[inpt]['filenames'][nrn]) as data_file:
							info=json.load(data_file)
						weights=np.array(info['weights'])
						locations=np.array(info['locations'])
						bioneuron.bias=info['bias']
						bioneuron.synapses[inpt]=np.empty((weights.shape[0],weights.shape[1]),dtype=object)
						bioneuron.netcons[inpt]=np.empty((weights.shape[0],weights.shape[1]),dtype=object)
						for n in range(weights.shape[0]):
							for s in range(weights.shape[1]):
								section=bioneuron.cell.apical(locations[n][s])
								weight=weights[n][s]
								synapse=ExpSyn(section,weight,self.P['tau'])
								bioneuron.synapses[inpt][n][s]=synapse
						bioneuron.start_recording()
			neuron.init()

		del decoder_model
		del decoder_sim

		return self.decoders, dict()