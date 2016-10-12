'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''

def make_hyp_params(P):
	import hyperopt
	n_syn=P['synapses_per_connection']
	n_LIF=P['n_LIF']
	hyp_params={'P':P,'weights':{},'locations':{},'bias':0}
	hyp_params['bias']=hyperopt.hp.uniform('b',P['bias_min'],P['bias_max'])
	for n in range(n_LIF):
		#adds a hyperopt-distributed weight, location, bias for each synapse
		for i in range(n_syn): 
			hyp_params['weights']['%s_%s'%(n,i)]=\
					hyperopt.hp.uniform('w_%s_%s'%(n,i),P['weight_min'],P['weight_max'])
			if P['synapse_dist'] == 'optimized': 
				hyp_params['locations']['%s_%s'%(n,i)]=\
					hyperopt.hp.uniform('l_%s_%s'%(n,i),0,1)
	return hyp_params

def tuning_curve_loss(P,LIFdata,X_NEURON,Hz_NEURON,hyp_params,addon):
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd
	import json
	import ipdb
	#shape of activities and Hz is mismatched, so interpolate and slice activities for comparison
	from scipy.interpolate import interp1d
	f_NEURON_rate = interp1d(X_NEURON,Hz_NEURON)
	f_LIF_rate = interp1d(np.array(LIFdata['X_LIF']),np.array(LIFdata['Hz_LIF']))
	x_min=np.maximum(LIFdata['X_LIF'][0],X_NEURON[0])
	x_max=np.minimum(LIFdata['X_LIF'][-1],X_NEURON[-1])
	X=np.arange(x_min,x_max,P['dx'])
	loss=np.sqrt(np.average((f_NEURON_rate(X)-f_LIF_rate(X))**2))
	sns.set(context='poster')

	figure, ax1 = plt.subplots(1,1)
	ax1.plot(X,f_NEURON_rate(X),label='bioneuron firing rate (Hz)')
	ax1.plot(X,f_LIF_rate(X),label='LIF firing rate (Hz)')
	ax1.set(xlabel='x',ylabel='firing rate (Hz)',title='loss=%0.3f' %loss)
	plt.legend()
	figure.savefig(hyp_params['directory']+'tuning_curve_%0.3f_'%loss + addon + '.png')
	plt.close(figure)
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
	from initialize import make_addon
	from analyze import get_rates, make_tuning_curves
	from neuron_methods import make_bioneuron, run_bioneuron
	import timeit

	start=timeit.default_timer()
	P=hyp_params['P']
	P['timesteps']=np.arange(0,P['t_sample'],P['dt'])
	n_syn=P['synapses_per_connection']
	n_LIF=P['n_LIF']
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
	run_neuron(P,LIFdata,bioneuron)
	addon=make_addon(6)
	rates=get_rates(P,bioneuron,LIFdata,addon,hyp_params)
	X_NEURON, Hz_NEURON = make_tuning_curves(P,LIFdata,rates)
	loss=tuning_curve_loss(P,LIFdata,X_NEURON,Hz_NEURON,hyp_params,addon)
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

	P=eval(open('parameters.txt').read())
	datadir=ch_dir()
	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	LIFdata=make_spikes_in(P,raw_signal,datadir)
	w_max=find_w_max(P,LIFdata)

	if P['optimization']=='hyperopt':
		hyp_params=make_hyp_params(P)
		hyp_params['directory']=datadir
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
		hyp_params=make_hyp_params(P)
		hyp_params['directory']=datadir
		trials=MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp3')
		best=hyperopt.fmin(simulate,space=hyp_params,algo=hyperopt.tpe.suggest,
							max_evals=P['max_evals'],trials=trials)
		plot_loss(trials,hyp_params)

if __name__=='__main__':
	main()