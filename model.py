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
	from analyze import get_rates, make_tuning_curves, tuning_curve_loss,export_biopop
	from analyze import	plot_rates, plot_tuning_curve
						
	from neuron_methods import make_bioneuron, run_bioneuron, connect_bioneuron
	import timeit

	start=timeit.default_timer()
	run_id=make_addon(6)
	os.chdir(P['directory'])

	start2=timeit.default_timer()
	lifdata=np.load(P['directory']+'lifdata.npz')
	signal_in=lifdata['signal_in']
	spikes_in=lifdata['spikes_in']
	lif_eval_points=lifdata['lif_eval_points'].ravel()
	lif_activities=lifdata['lif_activities']
	stop2=timeit.default_timer()
	print 'LIF data import/conversion - %s sec' %(stop2-start2)

	print 'Initializing bioneurons...'
	start4=timeit.default_timer()		
	biopop={}
	for b in range(P['n_bio']):
		bias=P['bias']['%s'%b]
		weights=np.zeros((P['n_lif'],P['n_syn']))
		locations=np.zeros((P['n_lif'],P['n_syn']))
		for n in range(P['n_lif']):
			for i in range(P['n_syn']):
				weights[n][i]=P['weights']['%s_%s_%s'%(b,n,i)]
				locations[n][i]=P['locations']['%s_%s_%s'%(b,n,i)]
		bioneuron = make_bioneuron(P,weights,locations,bias)
		connect_bioneuron(P,spikes_in,bioneuron)
		biopop[b]=bioneuron
	stop4=timeit.default_timer()
	print 'initialize - %s sec' % (stop4-start4)

	print 'Running NEURON...'
	start5=timeit.default_timer()		
	run_bioneuron(P)
	stop5=timeit.default_timer()
	print 'run - %s sec' % (stop5-start5)

	print 'Computing loss...'
	start6=timeit.default_timer()		
	all_spike_times=[]
	losses=[]
	for b in biopop.iterkeys():
		# start3=timeit.default_timer()
		spike_times=np.round(np.array(biopop[b].spikes),decimals=3)
		biospikes, biorates=get_rates(P,spike_times)
		# stop3=timeit.default_timer()
		# print 'get rates - %s sec' % (stop3-start3)	
		# start7=timeit.default_timer()
		bio_eval_points, bio_activities = make_tuning_curves(P,signal_in,biorates)
		# stop7=timeit.default_timer()
		# print 'make tuning curves - %s sec' % (stop7-start7)		
		X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(
				P,lif_eval_points,lif_activities[:,b],bio_eval_points,bio_activities)
		all_spike_times.append(spike_times)
		losses.append(loss)
		# plot_rates(P,loss,biopop[b],biospikes,biorates,signal_in,spikes_in,run_id)
		# plot_tuning_curve(P,X,f_bio_rate,f_lif_rate,loss,run_id)
	export_biopop(P,run_id,all_spike_times,losses)
	stop6=timeit.default_timer()
	print 'loss - %s sec' % (stop6-start6)
	stop=timeit.default_timer()
	print ' Total Runtime - %s sec' %(stop-start)
	return {'loss': np.sum(losses), 'run_id':run_id, 'status': hyperopt.STATUS_OK}


def main():
	import ipdb
	from initialize import ch_dir, make_signal, make_spikes_in, add_search_space
	from run_hyperopt import run_hyperopt
	from analyze import plot_loss, plot_biopop_tuning_curves #analyze_df

	P=eval(open('parameters.txt').read())
	datadir=ch_dir()
	P['directory']=datadir

	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	make_spikes_in(P,raw_signal)
	P=add_search_space(P)

	trials,best=run_hyperopt(P)
	plot_loss(P,trials)
	plot_biopop_tuning_curves(P,trials)

	# df=analyze_df(P,trials)

if __name__=='__main__':
	main()