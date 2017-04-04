import numpy as np
import pickle
import cPickle
import os

directory='GQYY38F8J'
ensemble='ens_bio'
n_neurons=30
os.chdir('/work/psipeter/bionengo/data/%s/%s/'%(directory,ensemble))

best_hyperparam_files=[]
for bionrn in range(n_neurons):
	print 'bioneuron_%s_hyperopt_trials.p'%bionrn
	trials=cPickle.load(open('bioneuron_%s_hyperopt_trials.p'%bionrn,'rb'))
	losses=[t['result']['loss'] for t in trials]
	ids=[t['result']['eval'] for t in trials]
	idx=np.argmin(losses)
	loss=np.min(losses)
	result=str(ids[idx])
	try:
		load_attempt=np.load('output_bioneuron_%s.npz'%bionrn)
		print 'load attempt losses', load_attempt['losses']
	except: #file doesn't exit
		print 'saving npz file for bioneuron %s'%bionrn
		np.savez('output_bioneuron_%s.npz'%bionrn,bionrn=bionrn,eval=result,losses=losses)
	best_hyperparam_files.append('/work/psipeter/bionengo/data/%s/%s/bioneuron_%s'%(directory,ensemble,bionrn))
np.savez('best_hyperparam_files.npz',best_hyperparam_files=best_hyperparam_files)

rates_bio, all_losses = [], []
for b in range(n_neurons):
	all_losses.append(np.load('output_bioneuron_%s.npz'%b)['losses'])
	rates_bio.append(np.load(best_hyperparam_files[b]+'/spikes_rates_bio_ideal.npz')['rates_bio'])
np.savez('target.npz',target=np.load('output_ideal_%s.npz'%ensemble)['values'])
np.savez('rates_bio.npz',rates_bio=np.array(rates_bio).T)
np.savez('all_losses.npz',losses=np.array(all_losses))

spikes_bio, spikes_ideal, rates_bio, rates_ideal=[],[],[],[]
for file in best_hyperparam_files:
	spikes_rates_bio_ideal=np.load(file+'/spikes_rates_bio_ideal.npz')
	spikes_bio.append(spikes_rates_bio_ideal['spikes_bio'])
	spikes_ideal.append(spikes_rates_bio_ideal['spikes_ideal'])
	rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
	rates_ideal.append(spikes_rates_bio_ideal['rates_ideal'])

spikes_bio=np.array(spikes_bio).T
spikes_ideal=np.array(spikes_ideal).T
rates_bio=np.array(rates_bio).T
rates_ideal=np.array(rates_ideal).T
print 'sum of rates_bio:', np.sum(rates_bio),rates_bio.shape
print 'sum of rates_ideal:', np.sum(rates_ideal),rates_ideal.shape
rmse=np.sqrt(np.average((rates_bio-rates_ideal)**2))
file=open('rmse_train.txt','w')
file.write(str(rmse))
file.close()
