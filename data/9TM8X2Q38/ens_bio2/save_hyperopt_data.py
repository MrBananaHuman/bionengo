import numpy as np
import pickle
import cPickle

for bionrn in range(30):
	print 'bioneuron_%s_hyperopt_trials.p'%bionrn
	trials=cPickle.load(open('bioneuron_%s_hyperopt_trials.p'%bionrn,'rb'))
	losses=[t['result']['loss'] for t in trials]
	ids=[t['result']['eval'] for t in trials]
	idx=np.argmin(losses)
	loss=np.min(losses)
	result=str(ids[idx])
	np.savez('output_bioneuron_%s_2.npz'%bionrn,bionrn=bionrn,eval=result,losses=losses)

for b in range(30):
	best_hyperparam_files, rates_bio2, best_losses, all_losses = [], [], [], []
	bionrn=np.load('output_bioneuron_%s_2.npz'%b)['bionrn']
	eval_number=np.load('output_bioneuron_%s_2.npz'%b)['eval']
	losses=np.load('output_bioneuron_%s_2.npz'%b)['losses']
	best_hyperparam_files.append('/work/psipeter/bionengo/data/9TM8X2Q38/ens_bio2/bioneuron_%s'%bionrn)
	spikes_rates_bio_ideal=np.load(best_hyperparam_files[-1]+'/spikes_rates_bio_ideal.npz')
	best_losses.append(np.load(best_hyperparam_files[-1]+'/loss.npz')['loss'])
	rates_bio2.append(spikes_rates_bio_ideal['rates_bio'])
	all_losses.append(losses)
        rates_bio2=np.array(rates_bio2).T
        target=np.load('output_ideal_ens_bio2.npz')['values']
        np.savez('best_hyperparam_files2.npz',best_hyperparam_files=best_hyperparam_files)
        np.savez('target2.npz',target=target)
	np.savez('rates_bio2.npz',rates_bio=rates_bio2)
	np.savez('all_losses2.npz',losses=np.array(all_losses))
	spikes_bio=[]
	spikes_ideal=[]
	rates_bio=[]
	rates_ideal=[]
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
	rmse=np.sqrt(np.average((rates_bio-rates_ideal)**2))
	file=open('rmse_train2.txt','w')
	file.write(str(rmse))
	file.close()
