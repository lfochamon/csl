# csl

Learning under requirements with pytorch

* Documentation: https://csl.readthedocs.io/en/latest/


## What is it?

**csl** (standing for *Constrained Statistical Learning*) is a Python package
based around pytorch to simplify the definition of constrained learning problems
and then solving them.

It was developed to run experiments for my [research](https://www.seas.upenn.edu/~luizf)
on learning under requirements.


## Requirements

* numpy
* pytorch
* matplotlib (for plotting)
* pandas (only for `csl.datasets`)
* PIL (only for `csl.datasets`)


## Installation

In your working folder simply do

```bash
   $ git clone https://github.com/lchamon/csl.git
```

or [download and extract](https://github.com/lchamon/csl/archive/main.zip).

If you use `conda`, you can set up a ready-to-go requirements by running

```bash
   $ conda env create -f environment.yml
   $ conda activate csl
```

**Note:** This environment uses `pytorch` without GPU support. If you need GPU support,
you should replace the package `cpuonly` in `environment.yml` with `cudatoolkit=XX.X`
where `XX.X` denotes your CUDA version.


## License
**csl** is distributed under the MIT license, see LICENSE.
