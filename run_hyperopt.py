def run_hyperopt(P):
	import hyperopt
	from model import simulate
	from analyze import plot_loss, get_min_loss_filename
	import numpy as np

	if P['hyperopt_algo'] == 'tpe':
		my_algo=hyperopt.tpe.suggest
	elif P['hyperopt_algo'] == 'random':
		my_algo=hyperopt.rand.suggest
	trials=hyperopt.Trials()
	best=hyperopt.fmin(simulate,
		space=P,
		algo=my_algo,
		max_evals=P['max_evals'],
		trials=trials)
	filename=get_min_loss_filename(P,trials)
	plot_loss(P,trials)

	return filename