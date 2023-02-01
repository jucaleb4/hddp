import os

# (0,1,2,3,4) EDDP, EDDPL, EDDPUL, SDDP, SEDDP 
mode = 1
sce_seed = 1111
sel_seed = 1
T = 120
niters = 3000
N = 120
debug = False
nprocs_arr = [1,2,4,6,8,10,12]

ct = 1100

for nprocs in nprocs_arr:
    print("--mode {} --niters {} --scen_seed {} --sel_seed {} --N {} --T {}".format(mode, niters, sce_seed, sel_seed, N, T))
    if debug:
        os.system("python main.py --mode {} --niters {} --scen_seed {} --sel_seed {} --N {} --T {} --prob 1 --nprocs {}".format(mode, niters, sce_seed, sel_seed, N, T, nprocs))
    else:
        os.system("python main.py --mode {} --niters {} --scen_seed {} --sel_seed {} --N {} --T {} --prob 1 --nprocs {} >> output/run_{}.txt".format(mode, niters, sce_seed, sel_seed, N, T, nprocs, ct))
    ct += 1
