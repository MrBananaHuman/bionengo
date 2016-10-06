'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''


def simulate(space):
	import numpy as np
	import neuron
	import hyperopt
	import json
	import ipdb
	from initialize import make_addon, make_bioneuron
	from analyze import get_rates, make_tuning_curves, calculate_loss
	from run import run_neuron

	P=space['P']
	P['timesteps']=np.arange(0,P['t_sample'],P['dt'])
	n_syn=P['synapses_per_connection']
	n_LIF=P['n_LIF']
	with open(space['directory']+"LIFdata.json") as data_file:  #linux  
		LIFdata = json.load(data_file)[0]
	weights=np.zeros((n_LIF,n_syn))
	locations=np.zeros((n_LIF,n_syn))
	bias=space['bias']
	for n in range(n_LIF):
		for i in range(n_syn):
			weights[n][i]=space['weights']['%s_%s'%(n,i)]
			if P['synapse_dist'] == 'optimized': 
				locations[n][i]=space['locations']['%s_%s'%(n,i)]
	bioneuron = make_bioneuron(P,weights,locations,bias)
	print '\nRunning NEURON'
	run_neuron(P,LIFdata,bioneuron)
	addon=make_addon(6)
	rates=get_rates(P,bioneuron,LIFdata,addon,space)
	X_NEURON, Hz_NEURON = make_tuning_curves(P,LIFdata,rates)
	loss=calculate_loss(P,LIFdata,X_NEURON,Hz_NEURON,space,addon)
	return {'loss': loss, 'status': hyperopt.STATUS_OK}

def main():
	import numpy as np
	import neuron
	import hyperopt
	import json
	import ipdb
	from initialize import ch_dir, make_search_space, make_signal, make_spikes_in
	from analyze import plot_loss

	P=eval(open('parameters.txt').read())
	datadir=ch_dir()
	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	LIFdata=make_spikes_in(P,raw_signal,datadir)

	if P['optimization']=='hyperopt':
		search_space=make_search_space(P)
		search_space['directory']=datadir
		trials=hyperopt.Trials()
		best=hyperopt.fmin(simulate,space=search_space,algo=hyperopt.tpe.suggest,
							max_evals=P['max_evals'],trials=trials)
		plot_loss(trials,search_space)

	elif P['optimization']=='mongodb':
		'''
		Commands to run MongoDB from Terminal
			https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB
		TAB 1: mongod --dbpath . --port 1234
		SET: exp_key='NEW_VALUE'
		TAB 2: python model.py
		TAB N: export PYTHONPATH=$PYTHONPATH:/home/pduggins/bionengo
		TAB N: hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1
		'''
		from hyperopt.mongoexp import MongoTrials
		search_space=make_search_space(P)
		search_space['directory']=datadir
		trials=MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp3')
		best=hyperopt.fmin(simulate,space=search_space,algo=hyperopt.tpe.suggest,
							max_evals=P['max_evals'],trials=trials)
		plot_loss(trials,search_space)

if __name__=='__main__':
	main()