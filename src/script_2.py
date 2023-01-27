import os

# (0,1,2,4) EDDP, EDDPL, EDDPUL, SEDDP 
modes = [4]
sce_seeds = [0]
sel_seeds = range(2,10)

niters = 3000

ct = 5

for sce_seed in sce_seeds:
    for sel_seed in sel_seeds:
        for mode in modes:
            print("ct={} mode={} scen_seed={} sel_seed={} niter={}".format(ct, mode, sce_seed, sel_seed, niters))
            os.system("python main.py --mode {} --niters {} --scen_seed {} --sel_seed {} >> output/run_{}.txt".format(mode, niters, sce_seed, sel_seed, ct))
            ct += 1
