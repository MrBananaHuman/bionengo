#!/usr/bin/env python
import numpy as np
import os
from bioneuron_helper import delete_extra_hyperparam_files
import sys
import json
def main():
    #create and save a list of the folder containing info for the best eval for each bioneuron
        param_name=sys.argv[1]
        with open(param_name,'r') as file:
                P=json.load(file)
	os.chdir(P['directory']+P['atrb']['label'])
        best_hyperparam_files, rates_bio, best_losses, all_losses = [], [], [], []
        for b in range(P['atrb']['neurons']):
                bionrn=np.load('output_bioneuron_%s.npz'%b)['bionrn']
                eval_number=np.load('output_bioneuron_%s.npz'%b)['eval']
                losses=np.load('output_bioneuron_%s.npz'%b)['losses']
                best_hyperparam_files.append(P['directory']+P['atrb']['label']+'/bioneuron_%s'%bionrn)
                # best_hyperparam_files.append(P['directory']+P['atrb']['label']+'/eval_%s_bioneuron_%s'%(eval_number,bionrn))
                spikes_rates_bio_ideal=np.load(best_hyperparam_files[-1]+'/spikes_rates_bio_ideal.npz')
                best_losses.append(np.load(best_hyperparam_files[-1]+'/loss.npz')['loss'])
                rates_bio.append(spikes_rates_bio_ideal['rates_bio'])
                all_losses.append(losses)
        rates_bio=np.array(rates_bio).T
        os.chdir(P['directory']+P['atrb']['label'])
        target=np.load('output_ideal_%s.npz'%P['atrb']['label'])['values']
        #delete files not in best_hyperparam_files
        delete_extra_hyperparam_files(P,best_hyperparam_files)
        #plot the spikes and rates of the best run
        #plot_spikes_rates_voltage_train(P,best_hyperparam_files,target,np.array(best_losses))
        #plot_hyperopt_loss(P,np.array(all_losses))
        np.savez('best_hyperparam_files.npz',best_hyperparam_files=best_hyperparam_files)
        np.savez('target.npz',target=target)
	np.savez('rates_bio.npz',rates_bio=rates_bio)
	np.savez('all_losses.npz',losses=np.array(all_losses))
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
	file=open('rmse_train.txt','w')
	file.write(str(rmse))
	file.close()
	return best_hyperparam_files,target,rates_bio

if __name__ == "__main__":
   main()
