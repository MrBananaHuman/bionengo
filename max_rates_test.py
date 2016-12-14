import nengo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb

def prime_sinusoids(t):
    return np.array([np.sin(t*2*np.pi*hz) for hz in [2,3]])

P={
	'dt_nengo':0.001,
	'ens_pre_neurons':30,
	'n_bio':30,
	'dim':2,
	'ens_pre_seed':222,
	'ens_ideal_seed':333,
	'min_rate':20,
	'max_rate':30,
	't_train':1.0,
	'nengo_synapse':0.05,
	'kernel':{'type':'gauss','sigma':0.05,},
}
with nengo.Network() as opt_model:
	signal = nengo.Node(lambda t: prime_sinusoids(t))
	pre = nengo.Ensemble(n_neurons=P['ens_pre_neurons'],
							dimensions=P['dim'],
							max_rates=nengo.dists.Uniform(P['min_rate'],P['max_rate']),
							seed=P['ens_pre_seed'])
	ideal = nengo.Ensemble(n_neurons=P['n_bio'],
							dimensions=P['dim'],
							max_rates=nengo.dists.Uniform(P['min_rate'],P['max_rate']),
							seed=P['ens_ideal_seed'])
	nengo.Connection(signal,pre,synapse=None)
	nengo.Connection(pre,ideal,synapse=P['nengo_synapse'])
	probe_signal = nengo.Probe(signal)
	probe_pre = nengo.Probe(pre,synapse=P['nengo_synapse'])
	probe_ideal = nengo.Probe(ideal.neurons,'spikes')

with nengo.Simulator(opt_model,dt=P['dt_nengo']) as opt_sim:
	opt_sim.run(P['t_train'])

input_signal=opt_sim.data[probe_signal]
pre_output=opt_sim.data[probe_pre]
spikes_ideal=opt_sim.data[probe_ideal]
spike_counts=[np.count_nonzero(spikes_ideal[:,i]) for i in range(spikes_ideal.shape[1])]
tkern = np.arange(-P['t_train']/20.0,P['t_train']/20.0,P['dt_nengo'])
kernel = np.exp(-tkern**2/(2*P['kernel']['sigma']**2))
kernel /= kernel.sum()
rates = np.array([np.convolve(kernel, spikes_ideal[:,i], mode='same') 
					for i in range(spikes_ideal.shape[1])]).T

print 'ens_ideal max_rates:', ideal.max_rates
print 'max spikes:', np.amax(spike_counts)
print 'max rates:', np.amax(rates)
sns.set(context='poster')
figure, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
ax1.plot(opt_sim.trange(),input_signal)
ax1.set(ylabel='input signal')
ax2.plot(opt_sim.trange(),pre_output)
ax2.set(ylabel='pre output')
ax3.plot(opt_sim.trange(),rates)
ax3.set(xlabel='time',ylabel='Hz')
plt.show()
figure.savefig('max_rates_test.png')