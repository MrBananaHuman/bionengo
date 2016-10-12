'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''

def make_hyp_params(P,w_max,datadir):
	import hyperopt
	n_syn=P['synapses_per_connection']
	n_LIF=P['n_LIF']
	hyp_params={'P':P,'weights':{},'locations':{},'bias':0,'directory':datadir}
	hyp_params['bias']=hyperopt.hp.uniform('b',P['bias_min'],P['bias_max'])
	for n in range(n_LIF):
		#adds a hyperopt-distributed weight, location, bias for each synapse
		for i in range(n_syn): 
			hyp_params['weights']['%s_%s'%(n,i)]=\
					hyperopt.hp.uniform('w_%s_%s'%(n,i),-1.0*w_max,1.0*w_max)
			if P['synapse_dist'] == 'optimized': 
				hyp_params['locations']['%s_%s'%(n,i)]=\
					hyperopt.hp.uniform('l_%s_%s'%(n,i),0,1)
	return hyp_params

def simulate(hyp_params):
	import numpy as np
	import neuron
	import hyperopt
	import json
	import ipdb
	from initialize import make_addon
	from analyze import get_rates, make_tuning_curves, tuning_curve_loss,\
						plot_rates, plot_tuning_curve,export_params
	from neuron_methods import make_bioneuron, run_bioneuron
	import timeit

	start=timeit.default_timer()
	P=hyp_params['P']
	n_syn=P['synapses_per_connection']
	n_LIF=P['n_LIF']
	run_id=make_addon(6)

	with open(hyp_params['directory']+"LIFdata.json") as data_file:  #linux  
		LIFdata = json.load(data_file)[0]
	weights=np.zeros((n_LIF,n_syn))
	locations=np.zeros((n_LIF,n_syn))
	bias=hyp_params['bias']
	for n in range(n_LIF):
		for i in range(n_syn):
			weights[n][i]=hyp_params['weights']['%s_%s'%(n,i)]
			if P['synapse_dist'] == 'optimized': 
				locations[n][i]=hyp_params['locations']['%s_%s'%(n,i)]
	bioneuron = make_bioneuron(P,weights,locations,bias)

	print '\nRunning NEURON'
	run_bioneuron(P,LIFdata,bioneuron)
	spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
	biospikes, biorates=get_rates(P,spike_times)
	bio_eval_points, bio_activities = make_tuning_curves(P,LIFdata,biorates)
	plot_rates(P,bioneuron,biospikes,biorates,LIFdata['signal_in'],LIFdata['spikes_in'],run_id)
	X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(P,LIFdata['lif_eval_points'],LIFdata['lif_activities'],bio_eval_points,bio_activities)
	plot_tuning_curve(X,f_bio_rate,f_lif_rate,loss,run_id)
	export_params(P,hyp_params,run_id)
	stop=timeit.default_timer()
	print ' Neuron Runtime - %s sec' %(stop-start)
	return {'loss': loss, 'status': hyperopt.STATUS_OK}


def main():
	import numpy as np
	import neuron
	import hyperopt
	import json
	import ipdb
	from initialize import ch_dir, make_signal, make_spikes_in, find_w_max
	from analyze import plot_loss

	P=eval(open('parameters.txt').read())
	datadir=ch_dir()
	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	LIFdata=make_spikes_in(P,raw_signal,datadir)
	w_max=find_w_max(P,LIFdata)
	print 'wmax=', w_max
	hyp_params=make_hyp_params(P,w_max,datadir)

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
		TAB 2: python model.py
		TAB N: export PYTHONPATH=$PYTHONPATH:/home/pduggins/bionengo
		TAB N: hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1
		'''
		from hyperopt.mongoexp import MongoTrials
		trials=MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp3')
		best=hyperopt.fmin(simulate,space=hyp_params,algo=hyperopt.tpe.suggest,
							max_evals=P['max_evals'],trials=trials)
		plot_loss(trials,hyp_params)

if __name__=='__main__':
	main()