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
						plot_rates, plot_tuning_curve,export_params, make_dataframe
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
	X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(P,lifdata['lif_eval_points'],lifdata['lif_activities'],bio_eval_points,bio_activities)
	P['loss']=loss

	os.chdir(P['directory'])
	plot_rates(P,bioneuron,biospikes,biorates,lifdata['signal_in'],lifdata['spikes_in'],run_id)
	plot_tuning_curve(P,X,f_bio_rate,f_lif_rate,loss,run_id)
	export_params(P,run_id,loss)
	df=make_dataframe(P,run_id,weights,locations,bias,loss)
	stop=timeit.default_timer()
	print ' Neuron Runtime - %s sec' %(stop-start)
	df.to_csv(run_id+'_dataframe')
	return {'loss': loss, 'run_id':run_id, 'status': hyperopt.STATUS_OK}


def main():
	import ipdb
	from initialize import ch_dir, make_signal, make_spikes_in, find_w_max, add_search_space
	from run_hyperopt import run_hyperopt
	from analyze import analyze_df

	P=eval(open('parameters.txt').read())
	datadir=ch_dir()
	P['directory']=datadir

	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	lifdata=make_spikes_in(P,raw_signal)
	# w_max=find_w_max(P,lifdata)
	P=add_search_space(P)

	trials=run_hyperopt(P)

	df=analyze_df(P,trials)

if __name__=='__main__':
	main()