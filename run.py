def run_neuron(P,LIFdata,bioneuron):
	import numpy as np
	import neuron
	import sys
	neuron.h.dt = P['dt'] * 1000
	neuron.init()
	for t in P['timesteps']: 
		sys.stdout.write("\r%d%%" %(1+100*t/(P['t_sample'])))
		sys.stdout.flush()
		for n in range(P['n_LIF']):
			# ipdb.set_trace()
			if np.any(np.where(np.array(LIFdata['spikes_in'])[:,n])[0] == int(t/P['dt'])):
				for syn in bioneuron.connections[n]:
					if syn.type=='ExpSyn': syn.conn.event(t*1000)
					elif syn.type=='Exp2Syn': syn.conn.event(t*1000)
					# elif syn.type=='AlphaSyn': syn.onset=t*1000 #TODO
		neuron.run(t*1000)