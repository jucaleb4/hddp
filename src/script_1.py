import os

# (0,1,2,4) EDDP, EDDPL, EDDPUL, SEDDP 
modes = [0,1,2,4]
seeds = [0]

niters = 3000

ct = 19

for seed in seeds:
    for mode in modes:
        print("ct={} mode={} seed={} niter={}".format(ct, mode, seed, niters))
        os.system("python main.py --mode {} --niters {} --scen_seed {} >> output/run_{}.txt".format(mode, niters, seed, ct))
        ct += 1
