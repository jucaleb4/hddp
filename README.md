# Hierarchal dual dynamic programming

This repository contains basic models, methodologies, and script for solving various infinite-horizon stochastic programs.
We consider a new generalization called *hierarchical stochastic programs*, where each stage's cost function is another (finite) multi-stage program.

In this documentation, we refer to EDDP as explorative dual dynamic programming and its generalization HDDP as hierarchial dual dynamic programming.

## Requirements
The code uses Python and has the basic requirements:
```
gurobipy==12.0.0
numpy==2.1.0
pandas==2.2.2
scipy==1.14.1
matplotlib==3.9.2
PyYAML==6.0.2
```

## Code structure
This section briefly discusses the major code components.

The code is contained in the `hddp` folder. The three main codes therein are `opt_setup.py`, `solver.py`, and `eddp.py`. 
The first file contains models for problems in the paper. The second contains solvers (e.g., Gurobi or a primal-dual method) for solving sub-problems that appear while running EDDP/HDDP. Finally, the third file holds the main code for running EDDP, which can solve infinite-horizon, possibly hierarchical, problems.

To run the code, scripts are automatically generated in the `script` folder.
The results are saved in the `log` folder.
How to use this scripts and run the code is described next.

## Creating scripts
In our code, scripts serve two purposes: 1) generate setting files needed to run the correct model and algorithm. 2) execute the code with the corresponding settings.

First, the settings files can be generated via
```
python scripts/2025_01_15/exp_0.py --setup 
```
and the settings file is saved in the `settings` folder. The flag `--setup` indicates we only want to create the settings file, not to execute code with it yet.

Different script files and the type of settings they run are given here:
- `2025_01_15/exp_0.py`: hydro-thermal planning with all EDDP and Inf-EDDP and variants
- `2025_01_16/exp_0.py`: inventory with all EDDP and Inf-EDDP and variants
- `2025_01_22/exp_0.py`: run hydro-thermal planning SDDP/EDDP longer (50K iterations, vs. previous 2K) for final optimality gap. 
- `2025_01_26/exp_0.py`: out-of-sample performance on inventory and hydro-thermal
- `2025_03_28/exp_0.py`: tuning for hierarchical inventory management
- `2025_03_28/exp_1.py`: tuning evaluation of hierarhcical inventory management
- `2025_03_30/exp_0.py`: out-of-sample performance of hueristics on hierarchical inventory 
- `2025_04_02/exp_0.py`: full run of PDSA on hierarchical inventory management (training)
- `2025_04_02/exp_1.py`: out-of-sample performance of PDSA on hierarchical inventory (testing)
- ~~`2025_04_03/exp_0.py`: tuning of hydropower generation planning~~ (not yet implemented)
- ~~`2025_04_03/exp_1.py`: evaluation of hydropower generation planning~~ (not yet implemented)
- `2025_04_23/exp_0.py`: risk-adverse inventory 

The reason we re-ran SDDP/EDDP in the experiments `2025_01_22/exp_0.py` (and not for the infinite-horizon counterparts) is because the non-inf methods require more oracle solves to achieve comparable performance to the infinite-horizon methods.

## Running scripts
After creating the settings file, the same script can be used to run the experiments, e.g.
```
python scripts/2025_01_15/exp_0.py --run
```
Here, you need to replace the data and experiment number to the one you want. 
Since running all experiments within a script can take a long time, you can run a subset of them using ``--sub_runs''. 
Use the `--help` command for more details.

## Plotting
See `/plots`.

## Citation
If you find this code useful, please cite our paper:
```
@misc{ju2023dualdynamicprogrammingstochastic,
      title={Dual dynamic programming for stochastic programs over an infinite horizon},
      author={Caleb Ju and Guanghui Lan},
      year={2023},
      eprint={2303.02024},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2303.02024},
}
```
