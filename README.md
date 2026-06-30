Sparse-PIM (SPIMA) is a functional simulator for in-DRAM based Sparse Matrix - Sparse Matrix Multiplication (SpMM) accelerator.

## Setup & Build

1. **Initialize Submodules** (Run from root repo):
```bash
source init.sh
```


2. **Build Simulator** (Run from gem5 folder):
```bash
cd gem5
chmod +x init.sh
./init.sh
```

### Run Simulation

```bash
build/X86/gem5.opt configs/deva.py -dim 2910 -Sp 97.94
```

### Modify pre-processing script
Go to `gem5` directory, and modify `main_old.c` script

Then compile
```bash
gcc main_old.c -o test
```

If you change binary name `test` to any other, go to `gem5/configs/` and modify `line 10` of `deva.py`
```bash
binary = "yournewbinary"
```

`gem5/configs/` also has `deva2.py`, which has extended memory mapping scheme. Modify/use it if needed.


### Note
Please do not commit to `main` branch, checkout to a new branch.