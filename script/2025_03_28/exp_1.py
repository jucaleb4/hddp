import os
import sys
import itertools
import argparse
import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from hddp import utils

MAX_RUNS = 72
DATE = "2025_03_28"
EXP_ID  = 1

def parse_sub_runs(sub_runs):
    start_run_id, end_run_id = 0, MAX_RUNS
    if (sub_runs is not None):
        try:
            start_run_id, end_run_id = sub_runs.split(",")
            start_run_id = int(start_run_id)
            end_run_id = int(end_run_id)
            assert 0 <= start_run_id <= end_run_id <= MAX_RUNS, "sub_runs id must be in [0,%s]" % (MAX_RUNS-1)
            
        except:
            raise Exception("Invalid sub_runs id. Must be two integers split between [0,%s] split by a single comma with no space" % (MAX_RUNS-1))

    return start_run_id, end_run_id

def setup_setting_files(seed_0, n_seeds, max_iter):
    od = dict([
        ('T', 128),
        ('N', 50),
        ('N2', 50),
        ('lam', 0.8),
        ('eps', 1e-3),
        ('max_iter', max_iter),
        ('policy_by_time', 200),
        ('eval_T', 24*10),
        ('prob_seed', seed_0),
        ('alg_seed', seed_0),
        ('mode', int(utils.Mode.INF_EDDP)),
        ('k1', 100),
        ('k2', 100),
        ('eta1_scale', 1.0),
        ('tau1_scale', 1.0),
        ('eta2_scale', 1.0),
        ('prob_name', 'hierarchical_inventory'),
        ('sa_eval', False),
        ('saa_eval', False),
    ])

    # number of tuning parameters in exp_0
    run_id_arr = list(range(72))
    eval_type_arr = ['sa_eval'] # ['sa_eval', 'saa_eval']
    prob_date_arr = [('hierarchical_inventory', '2025_03_28')]
    # first is in-sample, last 29 are out of sample
    prob_seed_N_arr = [(seed_0, 128)] # [(seed_0, 128)] + [(seed_0+i, od['T']*2) for i in range(1,30)]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not(os.path.exists(log_folder_base)):
        os.makedirs(log_folder_base)
    if not(os.path.exists(setting_folder_base)):
        os.makedirs(setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "eval"]
    row_format ="{:>10}|{:>10}"
    print("")
    print(row_format.format(*exp_metadata))
    print("-" * (21))

    ct = 0
    for (eval_type, run_id, (prob_name, date), (prob_seed, N)) in itertools.product(eval_type_arr, run_id_arr, prob_date_arr, prob_seed_N_arr):
        od[eval_type] = True
        od['prob_name'] = prob_name
        od['prob_seed'] = prob_seed
        od['N'] = N
        od['eval_fname'] = "eval_seed=%d.csv" % prob_seed

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        log_folder_base = os.path.join("logs", date, "exp_0")
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % run_id)
        assert os.path.exists(log_folder_base), "Folder %s does not exist, cannot run evaluation" % log_folder_base

        print(row_format.format(ct, eval_type))

        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(setting_fname, 'w') as f:
            # https://stackoverflow.com/questions/42518067/how-to-use-ordereddict-as-an-input-in-yaml-dump-or-yaml-safe-dump
            yaml.dump(od, f, default_flow_style=False, sort_keys=False)

        od[eval_type] = False
        ct += 1

    assert ct == MAX_RUNS, "Number of created exps (%i) does not match MAX_RUNS (%i)" % (ct, MAX_RUNS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument("--run", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument("--work", action="store_true", help="If true, only runs experiment for shor tperiod")
    parser.add_argument(
        "--sub_runs", 
        type=str, 
        help="Which experiments to run. Must be given as two integers separate by a comma with no space"
    )
    parser.add_argument("--parallel", action="store_true", help="Run seeds in parallel")

    args = parser.parse_args()
    seed_0 = 0
    n_seeds = 1
    max_iters = 10 if args.work else 50_000

    if args.setup:
        setup_setting_files(seed_0, n_seeds, max_iters)
    elif args.run:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", DATE, "exp_%i" % EXP_ID)

        for i in range(start_run_id, end_run_id):
            settings_file = os.path.join(folder_name, "run_%i.yaml" % i)
            os.system('echo "Running exp id %d"' % i)
            os.system("python main.py --settings %s --eval" % settings_file)
    else:
        print("Neither setup nor run passed. Shutting down...")
