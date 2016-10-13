'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''

def simulate(P):
	import numpy as np
	import neuron
	import hyperopt
	import json
	import ipdb
	import os
	from initialize import make_addon
	from analyze import get_rates, make_tuning_curves, tuning_curve_loss,\
						plot_rates, plot_tuning_curve,export_params
	from neuron_methods import make_bioneuron, run_bioneuron
	import timeit

	start=timeit.default_timer()
	run_id=make_addon(6)
	# if P['optimization']=='mongodb':
	# 	os.chdir("home/pduggins/bionengo/data/"+P['directory'])

	with open(P['directory']+"lifdata.json") as data_file:  #linux  
		lifdata = json.load(data_file)[0]
	weights=np.zeros((P['n_lif'],P['n_syn']))
	locations=np.zeros((P['n_lif'],P['n_syn']))
	bias=P['bias']
	for n in range(P['n_lif']):
		for i in range(P['n_syn']):
			weights[n][i]=P['weights']['%s_%s'%(n,i)]
			locations[n][i]=P['locations']['%s_%s'%(n,i)]
	bioneuron = make_bioneuron(P,weights,locations,bias)

	print '\nRunning NEURON'
	run_bioneuron(P,lifdata,bioneuron)
	spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
	biospikes, biorates=get_rates(P,spike_times)
	bio_eval_points, bio_activities = make_tuning_curves(P,lifdata,biorates)
	plot_rates(P,bioneuron,biospikes,biorates,lifdata['signal_in'],lifdata['spikes_in'],run_id)
	X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(P,lifdata['lif_eval_points'],lifdata['lif_activities'],bio_eval_points,bio_activities)
	plot_tuning_curve(X,f_bio_rate,f_lif_rate,loss,run_id)
	export_params(P,run_id)
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
	P['directory']=datadir
	P['bias']=hyperopt.hp.uniform('b',P['bias_min'],P['bias_max'])
	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	lifdata=make_spikes_in(P,raw_signal)
	w_max=find_w_max(P,lifdata)
	# w_max=0.01
	#adds a hyperopt-distributed weight, location, bias for each synapse
	P['weights']={}
	P['locations']={}
	for n in range(P['n_lif']):
		for i in range(P['n_syn']): 
			P['weights']['%s_%s'%(n,i)]=hyperopt.hp.uniform('w_%s_%s'%(n,i),-1.0*w_max,1.0*w_max)
			if P['synapse_dist'] == 'optimized': 
				P['locations']['%s_%s'%(n,i)]=hyperopt.hp.uniform('l_%s_%s'%(n,i),0,1)	
			elif P['synapse_dist'] == 'soma': 
				P['locations']['%s_%s'%(n,i)]=0.5
	if P['optimization']=='hyperopt':
		trials=hyperopt.Trials()
		best=hyperopt.fmin(simulate,space=P,algo=hyperopt.tpe.suggest,max_evals=P['max_evals'],trials=trials)
		plot_loss(P,trials)

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
		trials=MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
		best=hyperopt.fmin(simulate,space=P,algo=hyperopt.tpe.suggest,max_evals=P['max_evals'],trials=trials)
		plot_loss(P,trials)

if __name__=='__main__':
	main()