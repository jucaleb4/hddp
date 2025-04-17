import os
import sys
import itertools
import argparse
import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from hddp import utils

MAX_RUNS = 8
DATE = "2025_04_02"
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
        ('N', 50), # decrease to 25 to make it twice as faster
        ('N2', 50),
        ('lam', 0.9906),
        ('eps', 1e-3),
        ('max_iter', max_iter),
        ('time_limit', 1000),
        ('prob_seed', seed_0),
        ('alg_seed', seed_0),
        ('mode', int(utils.Mode.EDDP)),
        ('k1', 20),
        ('k2', 20),
        ('eta1_scale', 0.1),
        ('tau1_scale', 0.1),
        ('eta2_scale', 1.0),
        ('prob_name', 'hierarchical_inventory'),
        ('sa_eval', False),
        ('fixed_eval', False),
    ])

    lam_k1_k2_eta1_eta2_T_arr = [(0.8, 20, 20, 0.1, 1, 24), (0.9906, 100, 20, 1, 0.01, 128)]
    mode_seed_timelimit_arr = [
        (int(utils.Mode.INF_EDDP), 0, 1_000,), 
        (int(utils.Mode.GCE_INF_EDDP), 0, 1_000), 
        # (int(utils.Mode.INF_SDDP), 0),
        (int(utils.Mode.G_INF_SDDP), 0, 1_000), 
        (int(utils.Mode.SDDP), 0, 3_600),
    ]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not(os.path.exists(log_folder_base)):
        os.makedirs(log_folder_base)
    if not(os.path.exists(setting_folder_base)):
        os.makedirs(setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "lam", "mode", "alg_seed"]
    row_format ="{:>10}|" *(len(exp_metadata)-1) + "{:>10}"
    print("")
    print(row_format.format(*exp_metadata))
    print("-" * (11 * len(exp_metadata)))

    ct = 0
    od['h_bnds'] = [0, 500]
    for ((lam, k1, k2, eta1, eta2, T), (mode, seed, timelim)) in itertools.product(lam_k1_k2_eta1_eta2_T_arr, mode_seed_timelimit_arr):
        od["lam"] = lam
        od["k1"] = k1
        od["k2"] = k2
        od["eta1_scale"] = od["tau1_scale"] = eta1
        od["eta2_scale"] = eta2
        od["alg_seed"] = seed
        od["T"] = T
        od["mode"] = mode
        od["time_limit"] = timelim

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        print(row_format.format(ct, od["lam"], od["mode"], od["alg_seed"]))

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
    n_seeds = 1
    max_iters = 10 if args.work else 5_000

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
