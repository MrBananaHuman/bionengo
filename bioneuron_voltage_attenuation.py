'''
Peter Duggins
September 2016
'''

def weight_rescale(location,k):
	#interpolation
	import numpy as np
	from scipy.interpolate import interp1d
	#load voltage attenuation data for bahl.hoc
	voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
	f_voltage_att = interp1d(voltage_attenuation['distances'],voltage_attenuation['voltages'])
	# scaled_weight=1.0/f_voltage_att(location)
	scaled_weight=1.0/f_voltage_att(location)
	scaled_weight=scaled_weight*k
	return scaled_weight

	# #for nseg=5 in bahl.hoc
	# scaled_weight=1.0
	# if location == 0.0: scaled_weight/=1.0
	# elif 0.0 < location < 0.2: scaled_weight/=0.94
	# elif 0.2 <= location < 0.4: scaled_weight/=0.82
	# elif 0.4 <= location < 0.6: scaled_weight/=0.72
	# elif 0.6 <= location < 0.8: scaled_weight/=0.64
	# elif 0.8 <= location < 1.0: scaled_weight/=0.57
	# elif location == 1.0: scaled_weight/=0.54
	# return scaled_weight

def simulate(exp_params):
	import numpy as np
	from neuron_methods import run_bioneuron, make_bioneuron

	l0=exp_params[0]
	w0=exp_params[1]
	P=exp_params[2]

	LIFdata={}
	LIFdata['spikes_in']=np.zeros((int(P['t_sample']/P['dt']),1))
	LIFdata['spikes_in'][len(LIFdata['spikes_in'])/2]=1.0/P['dt'] #one spike in the middle

	print 'Running NEURON...'
	weights=np.ones((P['n_lif'],P['n_syn']))*w0
	weights*=weight_rescale(l0,P['k'])
	locations=np.ones((P['n_lif'],P['n_syn']))*l0
	bias=0.0
	bioneuron = make_bioneuron(P,weights,locations,bias)
	run_bioneuron(P,LIFdata,bioneuron)

	bio_v=np.array(bioneuron.v_record)
	midpoint_idx=int(len(bio_v)/2)-1
	v_rest=bio_v[midpoint_idx] #voltage before spike
	v_max=np.amax(bio_v[midpoint_idx:])
	dv=v_max-v_rest

	import matplotlib.pyplot as plt
	import seaborn as sns
	figure,ax=plt.subplots(1,1)
	ax.plot(bioneuron.t_record, bioneuron.v_record)
	ax.set(ylim=(-75,0))
	figure.savefig(P['directory']+'voltage_l0=%03d,w0=%06d.png'%(l0*100,w0*1e6))

	del bioneuron
	return [l0,w0,dv]

def main():
	import numpy as np
	from initialize import ch_dir
	from pathos.multiprocessing import ProcessingPool as Pool
	import matplotlib.pyplot as plt
	import seaborn as sns
	import ipdb

	P=eval(open('parameters.txt').read())
	P['timesteps']=np.arange(0,P['t_sample'],P['dt'])
	P['directory']=ch_dir()
	P['t_sample']=0.4
	# w_naughts=[0.01]
	w_naughts=np.linspace(0.0001,0.0002,num=50)
	l_naughts=[0.0]
	# l_naughts=np.linspace(0.0,1.0,num=5)
	P['n_lif']=1
	P['n_syn']=100
	P['k']=1.0
	n_processes=10


	pool = Pool(nodes=n_processes)
	exp_params=[]
	for l0 in l_naughts:
		for w0 in w_naughts:
			exp_params.append([l0, w0, P])
			# simulate([l0, w0, P])
	results=pool.map(simulate, exp_params)
	results=np.array(results)
	distances=results[:,0]
	weights=results[:,1]
	dvs=results[:,2]
	scaled_dvs=dvs/dvs[0]

	# np.savez('voltage_attenuation.npz',distances=distances,voltages=scaled_dvs)

	sns.set(context='poster')
	figure, ax1= plt.subplots(1, 1)
	plt.plot(distances,dvs/dvs[0])
	ax1.set(xlabel='distance from soma', ylabel='$\Delta V / \Delta V_{l=0}$',
			title='w0=%s_scaling_%s'%(weights[0],P['k']))
	figure.savefig(P['directory']+'voltage_attenuation_w0=%s_scaling_%s.png'
		%(weights[0],P['k']))


if __name__=='__main__':
	main()