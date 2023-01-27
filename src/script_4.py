import os

# (0,1,2,3,4) EDDP, EDDPL, EDDPUL, SDDP, SEDDP 
modes = [0,1]
sce_seeds = [0]
sel_seeds = range(1)

niters = 500
N = 10
debug = False

ct = 304

for sce_seed in sce_seeds:
    for sel_seed in sel_seeds:
        for mode in modes:
            print("ct={} mode={} scen_seed={} sel_seed={} N={} niter={}".format(ct, mode, sce_seed, sel_seed, N, niters))
            if debug:
                os.system("python main.py --mode {} --niters {} --scen_seed {} --sel_seed {} --N {}".format(mode, niters, sce_seed, sel_seed, N))
            else:
                os.system("python main.py --mode {} --niters {} --scen_seed {} --sel_seed {} --N {} >> output/run_{}.txt".format(mode, niters, sce_seed, sel_seed, N, ct))
            ct += 1
