'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''

def make_hyp_params(P):
	import hyperopt
	import numpy as np
	hyp_params={'P':P,'n_LIF':0,'weights':{},'locations':{},'bias':0}
	n_syn=P['synapses_per_connection']
	hyp_params['n_LIF']=np.random.randint(1,10) #hyperopt.hp.quniform('n_LIF',1,10,1)
	for n in range(hyp_params['n_LIF']):
		for i in range(n_syn): 
			hyp_params['weights']['%s_%s'%(n,i)]=hyperopt.hp.uniform('w_%s_%s'%(n,i),-1.0,1.0)
			hyp_params['bias']=hyperopt.hp.uniform('b',-1.0,1.0)
			hyp_params['locations']['%s_%s'%(n,i)]=0.5 #onto soma
	return hyp_params

def firing_rate_loss(P,Hz_NEURON,hyp_params,addon):
	import numpy as np
	import pandas as pd
	import json
	loss = -1.0*np.amax(Hz_NEURON)
	my_params=pd.DataFrame([hyp_params])
	my_params.reset_index().to_json(hyp_params['directory']+addon+\
						'_parameters_%0.3f_'%loss+'.json',orient='records')
	return loss

def simulate(hyp_params):
	import numpy as np
	import neuron
	import hyperopt
	import json
	import ipdb
	from initialize import make_addon, make_bioneuron
	from analyze import get_rates, make_tuning_curves
	from run import run_neuron
	import timeit

	start=timeit.default_timer()
	P=hyp_params['P']
	P['timesteps']=np.arange(0,P['t_sample'],P['dt'])
	n_syn=P['synapses_per_connection']
	n_LIF=hyp_params['n_LIF']
	with open(hyp_params['directory']+"LIFdata.json") as data_file:  #linux  
		LIFdata = json.load(data_file)[0]
	weights=np.zeros((n_LIF,n_syn))
	locations=np.zeros((n_LIF,n_syn))
	bias=hyp_params['bias']
	for n in range(n_LIF):
		for i in range(n_syn):
			weights[n][i]=hyp_params['weights']['%s_%s'%(n,i)]
			locations[n][i]=hyp_params['locations']['%s_%s'%(n,i)]
	bioneuron = make_bioneuron(P,weights,locations,bias)
	print '\nRunning NEURON'
	run_neuron(P,LIFdata,bioneuron)
	addon=make_addon(6)
	rates=get_rates(P,bioneuron,LIFdata,addon,hyp_params)
	X_NEURON, Hz_NEURON = make_tuning_curves(P,LIFdata,rates)
	loss=firing_rate_loss(P,Hz_NEURON,hyp_params,addon)
	print Hz_NEURON
	stop=timeit.default_timer()
	print ' Neuron Runtime - %s sec' %(stop-start)
	return {'loss': loss, 'status': hyperopt.STATUS_OK}


def main():
	import numpy as np
	import neuron
	import hyperopt
	import json
	import ipdb
	from initialize import ch_dir, make_signal, make_spikes_in
	from analyze import plot_loss
	from hyperopt.mongoexp import MongoTrials

	P=eval(open('parameters.txt').read())
	datadir=ch_dir()
	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	hyp_params=make_hyp_params(P)
	hyp_params['directory']=datadir
	P['n_LIF']=hyp_params['n_LIF']
	LIFdata=make_spikes_in(P,raw_signal,datadir)

	if P['optimization']=='hyperopt':
		trials=hyperopt.Trials()
		best=hyperopt.fmin(simulate,space=hyp_params,algo=hyperopt.tpe.suggest,
							max_evals=P['max_evals'],trials=trials)
		plot_loss(trials,hyp_params)

	elif P['optimization']=='mongodb':
		'''
		Commands to run MongoDB from Terminal
			https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB
		TAB 1: mongod --dbpath . --port 1234
		SET: exp_key='NEW_VALUE'
		TAB 2: python find_max_rates.py
		TAB N: export PYTHONPATH=$PYTHONPATH:/home/pduggins/bionengo
		TAB N: hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1
		'''
		trials=MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
		best=hyperopt.fmin(simulate,space=hyp_params,algo=hyperopt.tpe.suggest,
							max_evals=P['max_evals'],trials=trials)
		plot_loss(trials,hyp_params)

if __name__=='__main__':
	main()