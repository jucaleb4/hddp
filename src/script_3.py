import os

# (0,1,2,3,4) EDDP, EDDPL, EDDPUL, SDDP, SEDDP 
modes = [1]
probs = [2]
sce_seeds = [0]
sel_seeds = [0]
# errs = ["", "--bigerr"]
errs = [""]

niters = 1000

ct = 13
ct = 14 
ct = 15
ct = 400

for prob in probs:
    for sce_seed in sce_seeds:
        for sel_seed in sel_seeds:
            for mode in modes:
                for err in errs:
                    print("ct={} prob={} mode={} scen_seed={} sel_seed={} niter={} {}".format(ct, prob, mode, sce_seed, sel_seed, niters, err))
                    os.system("python main.py --prob {} --mode {} --niters {} --scen_seed {} --sel_seed {} {} >> output/run_{}.txt".format(
                        prob, mode, niters, sce_seed, sel_seed, err, ct))
                    ct += 1
