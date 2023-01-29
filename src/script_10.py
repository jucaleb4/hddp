import os

# (0,1,2,3,4) EDDP, EDDPL, EDDPUL, SDDP, SEDDP 
modes = [0,1,2,4]
sce_seeds = [1111]
sel_seeds = range(1,11)
Ts = [12, 120]

niters = 3000
N = 10
debug = False

ct = 1000

for T in Ts:
    for sce_seed in sce_seeds:
        for sel_seed in sel_seeds:
            for mode in modes:
                print("ct={} mode={} scen_seed={} sel_seed={} N={} niter={}".format(ct, mode, sce_seed, sel_seed, N, niters))
                if debug:
                    os.system("python main.py --mode {} --niters {} --scen_seed {} --sel_seed {} --N {} --prob 1".format(mode, niters, sce_seed, sel_seed, N))
                else:
                    os.system("python main.py --mode {} --niters {} --scen_seed {} --sel_seed {} --N {} --T {} >> output/run_{}.txt".format(mode, niters, sce_seed, sel_seed, N, T, ct))
                ct += 1
