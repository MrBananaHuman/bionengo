#!/usr/bin/env python
import numpy as np
import sys
from bioneuron_train import make_hyperopt_space_decomposed_weights_single_encoder,\
	make_hyperopt_space_decomposed_weights, make_hyperopt_space, run_hyperopt
import json
def main(): #called by bash_bioneuron_train.sh, results sent to OUTPUT_bioneuron_X.txt
        bionrn=int(sys.argv[1])
        param_name=sys.argv[2]
        with open(param_name,'r') as file:
                P=json.load(file)
        rng=np.random.RandomState(seed=P['hyperopt_seed']+P['atrb']['seed']+bionrn)
        if P['decompose_weights']==True:
                if P['single_encoder']==True: P_hyperopt=make_hyperopt_space_decomposed_weights_single_encoder(P,bionrn,rng)
                else: P_hyperopt=make_hyperopt_space_decomposed_weights(P,bionrn,rng)
        else: P_hyperopt=make_hyperopt_space(P,bionrn,rng)
        results=run_hyperopt(P_hyperopt)
        np.savez('output_bioneuron_%s.npz'%bionrn,bionrn=results[0],eval=results[1],losses=results[2])
        return results


if __name__ == "__main__":
   main()
