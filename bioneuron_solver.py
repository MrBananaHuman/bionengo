import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal
import copy
import ipdb
import json
import os
from synapses import ExpSyn, Exp2Syn
from tqdm import *

class BioneuronSolver(nengo.solvers.Solver):
	def __init__(self,P,ens,method):
		super(BioneuronSolver,self).__init__(weights=False)
		self.P=copy.copy(P)
		self.ens=ens
		self.method=method
		self.bahl_op=None
		self.activities=None
		self.target=None
		self.decoders=None
		self.solver=nengo.solvers.LstsqL2()

	def __call__(self,A,Y,rng=None,E=None):
		# print 'call', self.ens, self.ens.neuron_type, self.decoders
		from bioneuron_builder import BahlNeuron
		if self.decoders != None:
			return self.decoders, dict()
		elif isinstance(self.ens.neuron_type, BahlNeuron):
			if self.method == 'load':
				#load activities/targets from the runs performed during optimization
				print 'loading target signal and bioneuron activities for %s' %self.ens.neuron_type.atrb['label']
				self.targets=self.ens.neuron_type.father_op.targets
				self.activities=self.ens.neuron_type.father_op.activities
				min_len=np.min([len(self.targets), len(self.activities)])
				# ipdb.set_trace()
				self.decoders=self.solver(self.activities[:min_len,:],self.targets)[0]
				self.ens.neuron_type.father_op.decoders=self.decoders
			elif self.method == 'simulate':
				raise NotImplementedError
				#todo
			# self.decoders=np.ones(self.decoders.shape)
			# print 'decoders from bioneuron_solver, ens', self.ens, self.decoders
			return self.decoders, dict()
			
