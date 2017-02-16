import string
import random
import signals

def make_addon(N):
	addon=str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(N)))
	return addon

def ch_dir():
	#change directory for data and plot outputs
	root=os.getcwd()
	addon=make_addon(9)
	datadir=''
	if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
		datadir=root+'/data/'+addon+'/' #linux or mac
	elif sys.platform == "win32":
		datadir=root+'\\data\\'+addon+'\\' #windows
	os.makedirs(datadir)
	os.chdir(datadir) 
	return datadir

def make_signal(P):
	#todo:cleanup
	""" Returns: array indexed by t when called from a nengo Node"""
	signal_type=P['type']
	dt=P['dt']
	t_final=P['t_final']+dt #why?
	dim=P['dim']
	if signal_type =='equalpower':
		mean=P['mean']
		std=P['std']
		max_freq=P['max_freq']
		seed=P['seed']
		if seed == None:
			seed=np.random.randint(99999999)
	if signal_type == 'constant':
		mean=P['mean']
	if signal_type=='prime_sinusoids':
		raw_signal=signals.prime_sinusoids(dt,t_final,dim)
		return raw_signal
	raw_signal=[]
	for d in range(dim):
		if signal_type=='constant':
			raw_signal.append(signals.constant(dt,t_final,mean))
		elif signal_type=='white':
			raw_signal.append(signals.white(dt,t_final))
		elif signal_type=='white_binary':
			raw_signal.append(signals.white_binary(dt,t_final,mean,std))
		elif signal_type=='switch':
			raw_signal.append(signals.switch(dt,t_final,max_freq,))
		elif signal_type=='equalpower':
			raw_signal.append(signals.equalpower(dt,t_final,max_freq,mean,std,seed=seed))
		elif signal_type=='poisson_binary':
			raw_signal.append(signals.poisson_binary(dt,t_final,mean_freq,max_freq,low,high))
		elif signal_type=='poisson':
			raw_signal.append(signals.poisson(dt,t_final,mean_freq,max_freq))
		elif signal_type=='pink_noise':
			raw_signal.append(signals.pink_noise(dt,t_final,mean,std))
	assert len(raw_signal) > 0, "signal type not specified"
	#todo - scale to transform or radius
	return np.array(raw_signal)

def weight_rescale(location):
	#interpolation
	import numpy as np
	from scipy.interpolate import interp1d
	#load voltage attenuation data for bahl.hoc
	# voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
	voltage_attenuation=np.load('/home/pduggins/bionengo/'+'voltage_attenuation.npz')
	f_voltage_att = interp1d(voltage_attenuation['distances'],voltage_attenuation['voltages'])
	scaled_weight=1.0/f_voltage_att(location)
	return scaled_weight