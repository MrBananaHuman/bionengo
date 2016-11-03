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
	from analyze import get_rates, make_tuning_curves, tuning_curve_loss,export_bioneuron
	from analyze import	plot_rates, plot_tuning_curve					
	from neuron_methods import make_bioneuron, run_bioneuron, connect_bioneuron
	import timeit

	start=timeit.default_timer()
	run_id=make_addon(6)
	os.chdir(P['directory'])

	lifdata=np.load(P['directory']+'lifdata.npz')
	signal_in=lifdata['signal_in']
	spikes_in=lifdata['spikes_in']
	lif_eval_points=lifdata['lif_eval_points'].ravel()
	lif_activities=lifdata['lif_activities'][:,P['bio_idx']]

	# print 'Initializing bioneurons...'
	# start4=timeit.default_timer()		
	bias=P['bias']
	weights=np.zeros((P['n_lif'],P['n_syn']))
	locations=np.zeros((P['n_lif'],P['n_syn']))
	for n in range(P['n_lif']):
		for i in range(P['n_syn']):
			weights[n][i]=P['weights']['%s_%s'%(n,i)]
			locations[n][i]=P['locations']['%s_%s'%(n,i)]
	bioneuron = make_bioneuron(P,weights,locations,bias)
	connect_bioneuron(P,spikes_in,bioneuron)
	# stop4=timeit.default_timer()
	# print 'initialize - %s sec' % (stop4-start4)

	# print 'Running NEURON...'
	# start5=timeit.default_timer()		
	run_bioneuron(P)
	# stop5=timeit.default_timer()
	# print 'run - %s sec' % (stop5-start5)

	# print 'Computing loss...'
	# start6=timeit.default_timer()		
	spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
	# start3=timeit.default_timer()
	biospikes, biorates=get_rates(P,spike_times)
	# stop3=timeit.default_timer()
	# print 'get rates - %s sec' % (stop3-start3)	
	# start7=timeit.default_timer()
	bio_eval_points, bio_activities = make_tuning_curves(P,signal_in,biorates)
	# stop7=timeit.default_timer()
	# print 'make tuning curves - %s sec' % (stop7-start7)		
	X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(
			P,lif_eval_points,lif_activities,bio_eval_points,bio_activities)
	# plot_rates(P,loss,bioneuron,biospikes,biorates,signal_in,spikes_in,run_id)
	# plot_tuning_curve(P,X,f_bio_rate,f_lif_rate,loss,run_id)
	export_bioneuron(P,run_id,spike_times,loss)
	# stop6=timeit.default_timer()
	# print 'loss - %s sec' % (stop6-start6)
	del bioneuron
	stop=timeit.default_timer()
	# print 'Simulate Runtime - %s sec' %(stop-start)
	return {'loss': loss, 'run_id':run_id, 'status': hyperopt.STATUS_OK}


def main():
	import ipdb
	from initialize import ch_dir, make_signal, make_spikes_in, add_search_space
	from run_hyperopt import run_hyperopt
	from analyze import plot_final_tuning_curves
	from pathos.multiprocessing import ProcessingPool as Pool
	import copy
	import timeit
	import json

	main_start=timeit.default_timer()
	P=eval(open('parameters.txt').read())
	datadir=ch_dir()
	P['directory']=datadir

	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	make_spikes_in(P,raw_signal)

	P_list=[]
	pool = Pool(nodes=P['n_processes'])
	for bio_idx in range(P['n_bio']):
		P_idx=add_search_space(P,bio_idx)
		# run_hyperopt(P_idx)
		P_list.append(copy.copy(P_idx))

	filenames=pool.map(run_hyperopt, P_list)
	with open('filenames.txt','wb') as outfile:
		json.dump(filenames,outfile)

	total_loss=plot_final_tuning_curves(P,filenames)
	main_stop=timeit.default_timer()
	print '\nTotal Runtime - %s sec' %(main_stop-main_start)
	return total_loss

if __name__=='__main__':
	main()