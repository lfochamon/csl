:mod:`csl`
==========

.. automodule:: csl


Model wrappers
--------------

.. autoclass:: PytorchModel


Constrained learning problem
----------------------------

.. autoclass:: ConstrainedLearningProblem


Solvers
-------

.. autoclass:: PrimalThenDual
   :exclude-members: primal_dual_update

.. autoclass:: PrimalDual
   :exclude-members: primal_dual_update

.. autoclass:: SimultaneousPrimalDual
   :exclude-members: primal_dual_update


Base solver
-----------

.. autoclass:: csl.solver_base.SolverSettings

.. autoclass:: csl.solver_base.PrimalDualBase
