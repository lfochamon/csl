#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""csl: PyTorch-based constrained learning

The csl module provides a common interface to specify and
solve constrained learning problems using PyTorch.

"""

from csl.problem import ConstrainedLearningProblem

from csl.models import (
    PytorchModel,
)

from csl.solvers import (
    PrimalThenDual,
    SimultaneousPrimalDual,
    # Resilient,
)

# # Resilient versions
# ResilientPrimalThenDual = Resilient(PrimalThenDual)
# ResilientSimultaneousPrimalDual = Resilient(SimultaneousPrimalDual)

# Aliases
PrimalDual = PrimalThenDual
# ResilientPrimalDual = ResilientPrimalThenDual
