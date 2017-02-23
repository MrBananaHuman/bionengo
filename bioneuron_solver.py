import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal
import copy
import ipdb
import json
import os
from bioneuron_train import train_hyperparams
from synapses import ExpSyn, Exp2Syn
from tqdm import *
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
from bioneuron_helper import ch_dir, make_signal, load_spikes, load_hyperparams,\
		filter_spikes, compute_loss, plot_spikes_rates, plot_hyperopt_loss

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
		from bioneuron_builder import BahlNeuron
		if self.decoders != None:
			return self.decoders, dict()
		elif isinstance(self.ens.neuron_type, BahlNeuron):
			self.P['inpts']=self.ens.neuron_type.father_op.inputs
			self.P['atrb']=self.ens.neuron_type.father_op.ens_atributes
			#don't train if filenames already exists
			if self.ens.neuron_type.best_hyperparam_files != None and self.P['continue_optimization']==False:
				best_hyperparam_files, targets, activities = load_hyperparams(self.P)
			else:
				best_hyperparam_files, targets, activities = train_hyperparams(self.P)
			self.ens.neuron_type.father_op.best_hyperparam_files=best_hyperparam_files
			if self.method == 'load':
				#load activities/targets from the runs performed during optimization
				print 'loading target signal and bioneuron activities for %s' %self.P['atrb']['label']
				self.targets=targets
				self.activities=activities
				self.decoders=self.solver(self.activities,self.targets)[0]
			#elif self.method == 'simulate':
				#todo
			# print 'activities', self.activities
			# print 'target', self.targets
			# print 'decoders', self.decoders
			return self.decoders, dict()
		else:
			return nengo.solvers.LstsqL2()(A,Y,rng=rng,E=E)
			
