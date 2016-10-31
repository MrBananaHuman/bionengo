'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''

def weight_rescale(location):
	#interpolation
	import numpy as np
	from scipy.interpolate import interp1d
	#load voltage attenuation data for bahl.hoc
	voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
	f_voltage_att = interp1d(voltage_attenuation['distances'],voltage_attenuation['voltages'])
	scaled_weight=1.0/f_voltage_att(location)
	return scaled_weight

def simulate(exp_params):
	import numpy as np
	import neuron
	import nengo
	from nengo.utils.matplotlib import rasterplot
	from neuron_methods import run_bioneuron, make_bioneuron
	import matplotlib.pyplot as plt
	import seaborn as sns

	l0=exp_params[0]
	w0=exp_params[1]
	rates_in=exp_params[2]
	rates_out=exp_params[3]
	P=exp_params[4]

	for i in range(len(rates_in)):
		print '\nl0=%s, w0=%s, hz=%s' %(l0,w0,rates_in[i])
		hz=rates_in[i]
		P['min_lif_rate']=hz
		P['max_LIF_rate']=hz
		LIFdata={}
		with nengo.Network() as model:
			signal = nengo.Node(output=1.0)
			ens_in = nengo.Ensemble(P['n_lif'],
					dimensions=1,
					encoders=[[1]]*P['n_lif'],
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
		weights=np.ones((P['n_lif'],P['n_syn']))*w0
		if P['weight_attenuation']==True:
			weights*=weight_rescale(l0)
		locations=np.ones((P['n_lif'],P['n_syn']))*l0
		bias=0.0
		bioneuron = make_bioneuron(P,weights,locations,bias)
		run_bioneuron(P,LIFdata,bioneuron)

		timesteps=P['timesteps']
		spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
		spike_train=np.zeros_like(timesteps)
		for idx in spike_times/P['dt']/1000: 
			spike_train[idx]=1.0/P['dt']
		rates=np.zeros_like(spike_train)
		tkern = np.arange(-timesteps[-1]/4,timesteps[-1]/4,P['dt'])
		kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
		rates = np.convolve(kernel, spike_train, mode='same')
		sns.set(context='poster')
		figure, (ax1,ax2,ax3) = plt.subplots(3,1)
		ax1.plot(np.array(bioneuron.t_record)/1000, np.array(bioneuron.v_record))
		ax1.set( ylabel='voltage (mV)')
		rasterplot(timesteps, np.array(LIFdata['spikes_in']),ax=ax2,use_eventplot=True)
		ax2.set(ylabel='neuron')
		ax3.plot(timesteps,rates,label='output rate')
		ax3.set(xlabel='time (s)')
		plt.legend()
		newaddon=P['directory']+'hz'+str(hz)+'w'+str(w0)+'l'+str(l0)
		figure.savefig(newaddon+'_spikes.png')
		plt.close(figure)
		
		spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
		rates_out[i] = spike_times.shape[0] / P['t_sample']
		del bioneuron

	sns.set(context='poster')
	figure, ax1 = plt.subplots(1,1)
	ax1.plot(rates_in,rates_out)
	ax1.set(xlabel='LIF firing rate per input neuron', ylabel='bioneuron firing rate',ylim=(0,60),
			title='n_lif=%s, n_syn=%s, w0=%s'%(P['n_lif'],P['n_syn'],w0))
	figure.savefig(P['directory']+'response_curve_l0=%03d.png'%(l0*100))
	plt.close(figure)
	return

def main():
	import numpy as np
	from initialize import ch_dir
	from pathos.multiprocessing import ProcessingPool as Pool

	P=eval(open('parameters.txt').read())
	P['timesteps']=np.arange(0,P['t_sample'],P['dt'])
	P['directory']=ch_dir()
	rates_in=np.arange(25,500,25)
	rates_out=np.zeros_like(rates_in)
	w_naughts=[0.0001]
	# w_naughts=np.logspace(-3,-1,num=10)
	# l_naughts=[0.0]
	l_naughts=np.linspace(0.0,1.0,num=4)
	P['n_lif']=10
	P['n_syn']=5
	P['weight_attenuation']=True
	n_processes=8


	pool = Pool(nodes=n_processes)
	exp_params=[]
	for l0 in l_naughts:
		for w0 in w_naughts:
			exp_params.append([l0, w0, rates_in, rates_out, P])
			# simulate([l0, w0, rates_in, rates_out, P])
	pool.map(simulate, exp_params)

if __name__=='__main__':
	main()