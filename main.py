from hddp import eddp

# import gurobipy as gp
import numpy as np
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, required=True)
    parser.add_argument("--n_procs", default=1, type=int, help="Number of parallel processors")
    parser.add_argument("--eval", action="store_true", help="Eval mode")
    args = parser.parse_args()

    with open(args.settings) as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(0)

    if args.eval:
        eddp.HDDP_eval(settings)
    else:
        eddp.HDDP_multiproc(settings, args.n_procs)
