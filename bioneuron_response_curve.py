'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''


def main():
	import numpy as np
	import neuron
	import hyperopt
	import json
	import nengo
	import pandas as pd
	from initialize import ch_dir, make_addon
	from neuron_methods import run_bioneuron, make_bioneuron
	import matplotlib.pyplot as plt
	import seaborn as sns

	P=eval(open('parameters.txt').read())
	P['timesteps']=np.arange(0,P['t_sample'],P['dt'])
	datadir=ch_dir()
	P['directory']=datadir
	rates_in=np.arange(20,200,10)
	rates_out=np.zeros_like(rates_in)
	w_naughts=np.logspace(-2,-1,num=5)
	addon=make_addon(6)
	P['n_LIF']=1

	for w0 in w_naughts:
		for i in range(len(rates_in)):
			print '\nw0=%s, hz=%s' %(w0,rates_in[i])
			hz=rates_in[i]
			P['min_LIF_rate']=hz
			P['max_LIF_rate']=hz
			LIFdata={}
			with nengo.Network() as model:
				signal = nengo.Node(output=1.0)
				ens_in = nengo.Ensemble(P['n_LIF'],
						dimensions=1,
						encoders=[[1]],
						max_rates=nengo.dists.Uniform(hz,hz))
				nengo.Connection(signal,ens_in)
				probe_signal = nengo.Probe(signal)
				probe_in = nengo.Probe(ens_in.neurons,'spikes')
			print 'Generating input spikes...'
			with nengo.Simulator(model,dt=P['dt']) as sim:
				sim.run(P['t_sample'])
			signal_in=sim.data[probe_signal]
			spikes_in=sim.data[probe_in]
			LIFdata['signal_in']=signal_in.ravel()
			LIFdata['spikes_in']=spikes_in

			print 'Running NEURON...'
			n_syn=P['synapses_per_connection']
			weights=np.ones((P['n_LIF'],n_syn))*w0
			locations=np.ones((P['n_LIF'],n_syn))*0.5 #synapses onto where spikes are recorded on soma
			bias=-1.0
			bioneuron = make_bioneuron(P,weights,locations,bias)
			run_bioneuron(P,LIFdata,bioneuron)

			timesteps=P['timesteps']
			spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
			spike_train=np.zeros_like(timesteps)
			for idx in spike_times/P['dt']/1000: 
				spike_train[idx]=1.0
			rates=np.zeros_like(spike_train)
			tkern = np.arange(-timesteps[-1]/4,timesteps[-1]/4,P['dt'])
			kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
			rates = np.convolve(kernel, spike_train, mode='same')
			sns.set(context='poster')
			figure, (ax1,ax2) = plt.subplots(2,1)
			ax1.plot(np.array(bioneuron.t_record)/1000, np.array(bioneuron.v_record))
			ax1.set(xlabel='time', ylabel='voltage (mV)')
			ax2.plot(timesteps,LIFdata['signal_in'],label='input signal')
			for n in range(P['n_LIF']):
				ax2.plot(timesteps,np.array(LIFdata['spikes_in'])[:,n]*P['dt'],
					label='input spikes [%s]'%n)
			ax2.plot(timesteps,rates,label='output rate')
			ax2.set(xlabel='time (s)')
			plt.legend()
			newaddon='hz'+str(hz)+'w'+str(w0)
			figure.savefig(newaddon+'_spikes.png')
			plt.close(figure)
			
			spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
			rates_out[i] = spike_times.shape[0] / P['t_sample']
			del bioneuron

		sns.set(context='poster')
		figure, ax1 = plt.subplots(1,1)
		ax1.plot(rates_in,rates_out)
		ax1.set(xlabel='LIF firing rate', ylabel='bioneuron firing rate')
		figure.savefig(datadir+'w=%s'%w0+'response_curve.png')
		plt.close(figure)

if __name__=='__main__':
	main()