def run_hyperopt(P):
	import hyperopt
	from model import simulate
	import numpy as np

	if P['optimization']=='hyperopt':
		trials=hyperopt.Trials()
		best=hyperopt.fmin(simulate,space=P,algo=hyperopt.tpe.suggest,max_evals=P['max_evals'],trials=trials)

	elif P['optimization']=='mongodb':
		'''
		Commands to run MongoDB from Terminal
			https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB
		TAB 1: mongod --dbpath . --port 1234
		TAB 2: python model.py
		TAB N: export PYTHONPATH=$PYTHONPATH:/home/pduggins/bionengo
		TAB N: hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1
		'''
		from hyperopt.mongoexp import MongoTrials
		trials=MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key=str(np.random.randint(1e9)))
		best=hyperopt.fmin(simulate,space=P,algo=hyperopt.tpe.suggest,max_evals=P['max_evals'],trials=trials)
	

	return trials,best