import os
import sys
import itertools
import argparse
import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from hddp import utils

MAX_RUNS = 48
DATE = "2025_01_16"
EXP_ID  = 0

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
        ('lam', 0.99),
        ('eps', 1e-3),
        ('max_iter', max_iter),
        ('time_limit', 7200),
        ('prob_seed', seed_0),
        ('alg_seed', seed_0),
        ('mode', int(utils.Mode.INF_EDDP)),
        ('prob_name', 'hydro'),
    ])

    prob_name_arr = ['inventory']
    lam_n_iter_arr = [(0.99, 1_000, 128), (0.8, 100, 24)]
    mode_seed_arr = [
        (int(utils.Mode.INF_EDDP), 0), 
        (int(utils.Mode.CE_INF_EDDP), 0), 
        (int(utils.Mode.GAP_INF_EDDP), 0)
    ]  # 1024 so it does not converge too soon
    mode_seed_arr += list((int(utils.Mode.INF_SDDP), i)  for i in range(n_seeds))
    mode_seed_arr += [(int(utils.Mode.EDDP), 0)]
    mode_seed_arr += list((int(utils.Mode.SDDP), i)  for i in range(n_seeds))

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not(os.path.exists(log_folder_base)):
        os.makedirs(log_folder_base)
    if not(os.path.exists(setting_folder_base)):
        os.makedirs(setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "prob_name", "lam", "mode", "alg_seed"]
    row_format ="{:>10}|" * (len(exp_metadata)-1) + "{:>10}"
    print("")
    print(row_format.format(*exp_metadata))
    print("-" * ((len(exp_metadata)-1)*10 + 10 + len(exp_metadata)))

    ct = 0
    for (prob_name, (lam, n_iter, T), (mode, alg_seed)) in itertools.product(prob_name_arr, lam_n_iter_arr, mode_seed_arr):
        od["prob_name"] = prob_name
        od["lam"] = lam
        od["max_iter"] = min(max_iter, n_iter)
        od["mode"] = mode
        od["alg_seed"] = alg_seed
        od["T"] = 4*T if mode == int(utils.Mode.GAP_INF_EDDP) else min(T, od["max_iter"]) 

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        print(row_format.format(ct, od["prob_name"], od["lam"], od["mode"], od["alg_seed"]))

        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(setting_fname, 'w') as f:
            # https://stackoverflow.com/questions/42518067/how-to-use-ordereddict-as-an-input-in-yaml-dump-or-yaml-safe-dump
            yaml.dump(od, f, default_flow_style=False, sort_keys=False)
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
    n_seeds = 10
    max_iters = 10 if args.work else 1_000

    if args.setup:
        setup_setting_files(seed_0, n_seeds, max_iters)
    elif args.run:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", DATE, "exp_%i" % EXP_ID)

        for i in range(start_run_id, end_run_id):
            settings_file = os.path.join(folder_name, "run_%i.yaml" % i)
            os.system('echo "Running exp id %d"' % i)
            os.system("python main.py --settings %s" % settings_file)
    else:
        print("Neither setup nor run passed. Shutting down...")
