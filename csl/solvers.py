# -*- coding: utf-8 -*-
"""Constrained learning solvers

Provides different configurations of primal-dual updates and resilient versions.

"""

import numpy as np
import torch
from csl.solver_base import PrimalDualBase, SolverSettings
from csl.utils import _batches


class PrimalThenDual(PrimalDualBase):
    def __init__(self, user_settings={}):
        """Primal-then-dual solver.

        Update primal using a full pass over the dataset then
        update dual using a full pass over the dataset.

        Parameters
        ----------
        user_settings : `dict`, optional
            Dictionary containing solver settings. See `SolverSettings` for
            basic solver settings and defaults. Additional specific settings:
            * ``batch_size``: Mini-batch size. The default is `None` (uses full dataset at once).
            * ``shuffle``: Shuffle dataset before batching. The default is `True`.
            * ``dual_period``: Epoch period of dual update (update dual once every ``dual_period`` epochs).
            The default is 1, run once per primal epoch.

        """
        settings = SolverSettings({
            'batch_size': None,
            'shuffle': True,
            'dual_period': 1,
        })

        settings.initialize(user_settings)

        super().__init__(settings)


    def primal_dual_update(self, problem):
        ### PRIMAL ###
        primal_value_est, primal_grad_norm_est = self._primal(problem)

        ### DUAL ###
        if self.state_dict['HAS_CONSTRAINTS'] and self._every(self.settings['dual_period']):
            constraint_slacks_est, pointwise_slacks_est, dual_grad_norm_est = self._dual(problem)
        else:
            constraint_slacks_est, pointwise_slacks_est, dual_grad_norm_est = None, None, None

        return primal_value_est, primal_grad_norm_est, constraint_slacks_est, pointwise_slacks_est, dual_grad_norm_est


    def _primal(self, problem):
        primal_value_est = 0
        primal_grad_norm_est = 0

        if self.settings['shuffle']:
            idx_epoch = np.random.permutation(np.arange(problem.data_size))
        else:
            idx_epoch = range(0, problem.data_size)

        for batch_start, batch_end in _batches(problem.data_size, self.settings['batch_size']):
            batch_idx = idx_epoch[batch_start:batch_end]

            self.primal_solver.zero_grad()
            _, obj_value, _, _ = problem.lagrangian(batch_idx)
            self.primal_solver.step()

            with torch.no_grad():
                primal_value_est += obj_value*(batch_end - batch_start)/problem.data_size
                primal_grad_norm_est += np.sum([p.grad.norm().item()**2 for p in problem.model.parameters])*(batch_end - batch_start)/problem.data_size

        return primal_value_est, primal_grad_norm_est


    # Dual ascent step
    def _dual(self, problem):
        constraint_slacks, pointwise_slacks = problem.slacks()

        # Update gradients
        dual_grad_norm = 0
        for ii, slack in enumerate(constraint_slacks):
            problem.lambdas[ii].grad = -slack
            if problem.lambdas[ii] > 0 or (problem.lambdas[ii] == 0 and slack > 0):
                dual_grad_norm += slack.item()**2

        for ii, slack in enumerate(pointwise_slacks):
            problem.mus[ii].grad = -slack
            inactive = torch.logical_or(problem.mus[ii] > 0, \
                                        torch.logical_and(problem.mus[ii] == 0, slack > 0))
            dual_grad_norm += torch.norm(slack[inactive]).item()**2

        # Take gradient step
        self.dual_solver.step()

        # Project onto non-negative orthant
        for ii, _ in enumerate(problem.lambdas):
            problem.lambdas[ii][problem.lambdas[ii] < 0] = 0
        for ii, _ in enumerate(problem.mus):
            problem.mus[ii][problem.mus[ii] < 0] = 0

        return constraint_slacks, pointwise_slacks, dual_grad_norm



class SimultaneousPrimalDual(PrimalDualBase):
    def __init__(self, user_settings={}):
        """Simultaneous primal-dual solver.

        For each batch, update primal then update dual.

        Parameters
        ----------
        user_settings : `dict`, optional
            Dictionary containing solver settings. See `SolverSettings` for
            basic solver settings and defaults. Additional specific settings:
            * ``batch_size``: Mini-batch size. The default is `None` (uses full dataset at once).
            * ``shuffle``: Shuffle dataset before batching. The default is `True`.

        """
        settings = SolverSettings({
            'batch_size': None,
            'shuffle': True
        })

        settings.initialize(user_settings)

        super().__init__(settings)


    def primal_dual_update(self, problem):
        # Initialize estimates
        primal_value_est = 0
        primal_grad_norm_est = 0
        if self.state_dict['HAS_CONSTRAINTS']:
            constraint_slacks_est = [torch.tensor(0, dtype = torch.float,
                                             requires_grad = False,
                                             device = self.settings['device']) \
                                    for _ in problem.rhs]
            pointwise_slacks_est = [torch.zeros_like(rhs, dtype = torch.float,
                                             requires_grad = False,
                                             device = self.settings['device']) \
                                   for rhs in problem.pointwise_rhs]
            dual_grad_norm_est = 0
        else:
            constraint_slacks_est, pointwise_slacks_est, dual_grad_norm_est = None, None, None

        # Shuffle dataset
        if self.settings['shuffle']:
            idx_epoch = np.random.permutation(np.arange(problem.data_size))
        else:
            idx_epoch = range(0, problem.data_size)

        ### START OF EPOCH ###
        for batch_start, batch_end in _batches(problem.data_size, self.settings['batch_size']):
            batch_idx = idx_epoch[batch_start:batch_end]

            ### PRIMAL UPDATE ###
            # Gradient step
            self.primal_solver.zero_grad()
            _, obj_value, constraint_slacks, pointwise_slacks = problem.lagrangian(batch_idx)
            self.primal_solver.step()

            # Compute primal quantities estimates
            with torch.no_grad():
                primal_value_est += obj_value*(batch_end - batch_start)/problem.data_size
                primal_grad_norm_est += np.sum([p.grad.norm().item()**2 for p in problem.model.parameters])*(batch_end - batch_start)/problem.data_size

            ### DUAL UPDATE ###
            if self.state_dict['HAS_CONSTRAINTS']:
                # Set gradients
                for ii, slack in enumerate(constraint_slacks):
                    problem.lambdas[ii].grad = -slack
                    constraint_slacks_est[ii] += slack*(batch_end - batch_start)/problem.data_size

                    if problem.lambdas[ii] > 0 or (problem.lambdas[ii] == 0 and slack > 0):
                        dual_grad_norm_est += slack**2*(batch_end - batch_start)/problem.data_size

                for ii, slack in enumerate(pointwise_slacks):
                    expanded_slack = torch.zeros_like(problem.mus[ii])
                    expanded_slack[batch_idx] = slack
                    problem.mus[ii].grad = -expanded_slack
                    pointwise_slacks_est[ii][batch_idx] = slack

                    inactive = torch.logical_or(problem.mus[ii][batch_idx] > 0, \
                                                torch.logical_and(problem.mus[ii][batch_idx] == 0, slack > 0))
                    dual_grad_norm_est += torch.norm(slack[inactive]).item()**2

                # Gradient gradient step
                self.dual_solver.step()

                # Project onto non-negative orthant
                for ii, _ in enumerate(problem.lambdas):
                    problem.lambdas[ii][problem.lambdas[ii] < 0] = 0
                for ii, _ in enumerate(problem.mus):
                    problem.mus[ii][problem.mus[ii] < 0] = 0

        return primal_value_est, primal_grad_norm_est, constraint_slacks_est, pointwise_slacks_est, dual_grad_norm_est
