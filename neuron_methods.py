#if building new NEURON mechanisms (channel, synapse)
	#>>> cd folder_with_file.mod
	#>>> /usr/local/x86_64/bin/nrnivmodl
import nengo
import neuron
import numpy as np

class BioneuronNode(nengo.Node):
	
	def __init__(self,n_in,n_bio,n_syn,dt=0.0001,tau=0.01,synapse_type='ExpSyn',preoptimized=True):
		super(BioneuronNode,self).__init__(self.step,size_in=n_in,size_out=n_bio)
		self.n_in=n_in
		self.n_bio=n_bio
		self.n_syn=n_syn
		self.tau=tau
		self.syn_type=synapse_type
		if preoptimized == True:
			self.biopop=self.load_biopop()
		else:
			self.biopop=self.optimize_biopop()
		neuron.h.dt = dt*1000
		neuron.init()

	def load_biopop(self):
		import json
		import numpy as np
		from neurons import Bahl
		datadir='data/8RCOQKGYQ/data/XUVLD1EGK/'
		directory='/home/pduggins/bionengo/'+datadir
		f=open(directory+'filenames.txt','r')
		filenames=json.load(f)
		biopop=[]
		for bio_idx in range(self.n_bio):
			bioneuron=Bahl()
			with open(filenames[bio_idx],'r') as data_file: 
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
		raise NotImplementedError()

	def step(self,t,x):
		#TODO: relax assumption that dt_nengo = dt_neuron
		#x is an array, size n_in, of whether input neurons spiked at time=t
		for n in range(self.n_in):
			if x[n] > 0:
				#for all bioneurons, add a spike to all synapses connected to input neuron
				for bioneuron in self.biopop:
					for syn in bioneuron.synapses[n]:
						syn.spike_in.event(1.0*t*1000) #convert from s to ms
		neuron.run(neuron.h.t + neuron.h.dt)
		output=[]
		for bioneuron in self.biopop:
			spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
			if len(spike_times) == 0:
				output.append(0)
			elif 1.0*spike_times[-1]/1000>(t-1.0*neuron.h.dt/1000):
				output.append(1)
			else:
				output.append(0)
		return output

def make_bioneuron(P,weights,locations,bias):
	import numpy as np
	from neurons import Bahl
	bioneuron=Bahl()
	#make connections and populate with synapses
	for n in range(P['n_lif']):
		bioneuron.add_bias(bias)
		bioneuron.add_connection(n)
		for i in range(P['n_syn']):
			syn_type=P['synapse_type']
			if P['synapse_dist'] == 'soma': section=bioneuron.cell.soma(locations[n][i])
			elif P['synapse_dist'] == 'tuft': section=bioneuron.cell.tuft(locations[n][i])
			else: section=bioneuron.cell.apical(locations[n][i])
			weight=weights[n][i]
			tau=P['synapse_tau']
			tau2=P['synapse_tau2']
			bioneuron.add_synapse(n,syn_type,section,weight,tau,tau2)
	#initialize recording attributes
	bioneuron.start_recording()
	return bioneuron

def connect_bioneuron(P,spikes_in,bioneuron):
	import numpy as np
	import neuron
	import ipdb
	for n in range(P['n_lif']):
		#create spike time vectors and an artificial spiking cell that delivers them
		vstim=neuron.h.VecStim()
		bioneuron.vecstim[n]['vstim'].append(vstim)
		spike_times_ms=list(1000*P['dt']*np.nonzero(spikes_in[:,n])[0]) #timely
		vtimes=neuron.h.Vector(spike_times_ms)
		bioneuron.vecstim[n]['vtimes'].append(vtimes)
		bioneuron.vecstim[n]['vstim'][-1].play(bioneuron.vecstim[n]['vtimes'][-1])
		#connect the VecStim to each synapse
		for syn in bioneuron.synapses[n]:
			netcon=neuron.h.NetCon(bioneuron.vecstim[n]['vstim'][-1],syn.syn)
			netcon.weight[0]=abs(syn.weight)
			bioneuron.netcons[n].append(netcon)

def run_bioneuron(P):
	import neuron
	neuron.h.dt = P['dt']*1000
	neuron.init()
	neuron.run(P['t_sample']*1000)