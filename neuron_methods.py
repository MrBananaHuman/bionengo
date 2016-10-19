#if building new NEURON mechanisms (channel, synapse)
	#>>> cd folder_with_file.mod
	#>>> /usr/local/x86_64/bin/nrnivmodl
	
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
			else: section=bioneuron.cell.apical(locations[n][i])
			weight=weights[n][i]
			tau=P['synapse_tau']
			tau2=P['synapse_tau2']
			bioneuron.add_synapse(n,syn_type,section,weight,tau,tau2)
	#initialize recording attributes
	bioneuron.start_recording()
	return bioneuron

def run_bioneuron(P,LIFdata,bioneuron):
	import numpy as np
	import neuron
	import sys
	timesteps=np.arange(0,P['t_sample'],P['dt'])
	neuron.h.dt = P['dt'] * 1000
	neuron.init()
	for t in timesteps: 
		sys.stdout.write("\r%d%%" %(1+100*t/(P['t_sample'])))
		sys.stdout.flush()
		t_idx=np.rint(t/P['dt'])
		t_neuron=t*1000
		for n in range(P['n_lif']):
			if np.any(np.where(np.array(LIFdata['spikes_in'])[:,n])[0] == t_idx):
				for syn in bioneuron.connections[n]:
					if syn.type=='ExpSyn': syn.conn.event(t_neuron)
					elif syn.type=='Exp2Syn': syn.conn.event(t_neuron)
					# elif syn.type=='AlphaSyn': syn.onset=t*1000 #TODO
		neuron.run(t_neuron)