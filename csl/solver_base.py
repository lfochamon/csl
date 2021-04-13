# -*- coding: utf-8 -*-
"""Base constrained learning solver

Base primal-dual solver.

"""


import numpy as np
import torch
import matplotlib.pyplot as plot
import logging



class SolverSettings():
    """Primal-dual solver settings

    Attributes
    ----------
    settings : `dict`, optional
        Dictionary containing the solver settings. The base solver valid
        keys and default values are listed below. Specific solvers may
        have additional settings.

        - ``iterations``: Maximum number of iterations. The default is 100.
        - ``primal_solver``: Constructor for the PyTorch solver used to solve the primal problem
          (takes parameter, returns torch.optim). The default is ADAM.
        - ``lr_p_scheduler``: Primal step size scheduler. The default is `None` (no decay).
        - ``dual_solver``: Constructor for the PyTorch solver used to solve the dual problem
          (takes parameter, returns torch.optim). The default is ADAM.
        - ``lr_d_scheduler``: Dual step size scheduler. The default is `None` (no decay).
        - ``logger``: A `logging` object. The default outputs directly to the console.
        - ``verbose``: Period of log printing. Every iteration is printed when logger is at DEBUG level.
          Default is ``iterations/10``. Set to 0 to deactivate.
        - ``device``: Device used for computations. The default is GPU, if available, and CPU otherwise.
        - ``COMPUTE_TRUE_DGAP``: Compute actual primal and dual values instead of using approximations.
          Note: this effectively requires an extra pass over the whole dataset per iteration.
          The default is `False`.
        - ``STOP_DIVERGENCE``: Maximum value allowed before declaring that
          the algorithm has diverged. The default is 1e4.
        - ``STOP_PVAL``: Primal value threshold. The default is `None`.
        - ``STOP_PGRAD``: Primal gradient squared norm threshold. The default is `None`.
        - ``STOP_ABS_DGAP``: Absolute duality gap threshold. The default is `None`.
        - ``STOP_DGRAD``: Dual gradient squared norm threshold. The default is `None`.
        - ``STOP_REL_DGAP``: Relative duality gap threshold. The default is `None`.
        - ``STOP_ABS_FEAS``: Absolute feasibility threshold. The default is `None`.
        - ``STOP_REL_FEAS``: Relative feasibility threshold. The default is `None`.
        - ``STOP_NFEAS``: Proportion of feasible constraints threshold. The default is `None`.
        - ``STOP_PATIENCE``: Iterations with no update threshold. The default is `None`.
        - ``STOP_USER_DEFINED``: User-defined function. Takes problem
          and the solver state_dict and returns `True` to stop or
          `False` to continue. The default is `None`.

    Raises
    ------
    `ValueError`
        When trying to get or set a non-existant setting

    """

    def __init__(self, specific_settings={}):
        """Primal-dual solver settings constructor

        Parameters
        ----------
        specific_settings : `dict`, optional
            Solver-specific global settings with default values.
            The default is ``{}``.

        """
        # Default global settings
        self.global_settings = {
            'iterations': 100,
            'verbose': None,
            'primal_solver': torch.optim.Adam,
            'lr_p_scheduler': None,
            'dual_solver': torch.optim.Adam,
            'lr_d_scheduler': None,
            'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'logger': None,
            'STOP_DIVERGENCE': 1e4,
            'COMPUTE_TRUE_DGAP': False,
            'STOP_PVAL': None,
            'STOP_PGRAD': None,
            'STOP_ABS_DGAP': None,
            'STOP_DGRAD': None,
            'STOP_REL_DGAP': None,
            'STOP_ABS_FEAS': None,
            'STOP_REL_FEAS': None,
            'STOP_NFEAS': None,
            'STOP_PATIENCE': None,
            'STOP_USER_DEFINED': None
        }

        # Update default global settings
        for key, value in specific_settings.items():
            self.global_settings[key] = value

        # Local settings
        self.local_settings = {}

    def initialize(self, settings={}):
        """Initialize settings

        Define global settings and initialize variable settings
        if not defined (namely, ``verbose`` and ``logger``).

        Parameters
        ----------
        settings : `dict`, optional
            User settings to override the default global settings.
            The default is ``{}``.

        """
        # Override global defaults with user inputs
        for key, value in settings.items():
            if self._setting_exists(key):
                self.global_settings[key] = value

        # Default logging settings
        if self.global_settings['verbose'] is None:
            self.global_settings['verbose'] = self.global_settings['iterations']/10

        if self.global_settings['logger'] is None:
            self.global_settings['logger'] = logging.getLogger('csl')

            # Add stream handler (console), if not defined already
            if not self.global_settings['logger'].hasHandlers():
                self.global_settings['logger'].addHandler(logging.StreamHandler())

            # Set level to info
            self.global_settings['logger'].setLevel(logging.INFO)

    def override(self, local_settings):
        """Mask global setting values

        Override the global settings without modifying them.

        Parameters
        ----------
        local_settings : `dict`
            Setting values to override.

        """
        for key in local_settings.keys():
            self._setting_exists(key)

        self.local_settings = local_settings

    def display(self):
        """Display effective setting values

        """
        max_length = np.max([len(key) for key in self.global_settings])

        for key in self.global_settings:
            if key in self.local_settings:
                print(key + ':' + ' '*(max_length - len(key) + 1) + f'{self.local_settings[key]}' + f' (global: {self.global_settings[key]})')
            elif key in self.global_settings:
                print(key + ':' + ' '*(max_length - len(key) + 1) + f'{self.global_settings[key]}')

    def __getitem__(self, key):
        if self._setting_exists(key):
            return self.local_settings.get(key) or self.global_settings.get(key)

    def __setitem__(self, key, value):
        self.global_settings[key] = value

    def _setting_exists(self, key):
        if key in self.global_settings:
            return True
        else:
            raise ValueError(f'Unknown setting "{key}"')



class PrimalDualBase():
    """Primal-dual base solver

    Attributes
    ----------
        primal_solver : `torch.optim`
            Primal problem solver
        primal_step_size : `torch.optim.lr_scheduler`
            Primal problem step size scheduler
        dual_solver : `torch.optim`
            Dual problem solver
        dual_step_size : `torch.optim.lr_scheduler`
            Dual problem step size scheduler
        state_dict : `dict`
            Internal solver state
        settings : `SolverSettings`
            Solver settings

    Notes
    -----

    **Stopping criteria**

    By default, the solver stops only once it reaches the maximum
    number of iterations or if divergence is detected (using the
    threshold ``STOP_DIVERGENCE``). When defined, other stopping
    modes are:

    - Primal absolute optimality: based on ``STOP_PVAL`` or ``STOP_PGRAD``
      (either or both depending on whether exist). Applies only to
      unconstrained problems.
    - Primal absolute optimality and absolute feasibility: as above
      but also check ``STOP_ABS_FEAS`` (on average constraints) and
      ``STOP_NFEAS`` (for pointwise constraints), depending on whether
      they exist.
    - Primal-dual absolute optimality and absolute feasibility:
      checks optimality using ``STOP_ABS_DGAP`` or both ``STOP_PGRAD`` and ``STOP_DGRAD``.
      Feasibility is checked using ``STOP_ABS_FEAS`` (on average constraints) and
      ``STOP_NFEAS`` (for pointwise constraints), depending on whether they exist.
    - Primal-dual relative optimality and relative feasibility:
      checks optimality using ``STOP_REL_DGAP`` and feasibility using
      ``STOP_REL_FEAS`` (for average constraints) and ``STOP_NFEAS``
      (for pointwise constraints).
    - Stalled: based on whether neither primal value nor any
      constraint violation has improved over the span of
      ``STOP_PATIENCE`` iterations.
    - User-defined criterion: stops if ``STOP_USER_DEFINED`` returns ``True``.


    **Implementing a new solver**

    Subclasses must:

    - Call ``PrimalDualBase.__init__`` with a `SolverSettings` object that
      includes its solver-specific settings (if any).
    - Define ``primal_dual_update``


    **Solver states**

    Specific solvers (and users) are free to modify and add to ``state_dict``.
    This can be useful to save internal states of the user-defined stopping
    criterion ``STOP_USER_DEFINED``, which can also be used as a validation hook.
    You should use a unique capitalized prefix in order to avoid interfering with
    the normal operation of the primal-dual solver (unless you know what you are doing).
    ``PrimalDualBase`` has the following internal states:

    - ``iteration`` (`int`): Iteration number
    - ``no_update_iterations`` (`int`): Number of iterations without updates left
      until solver gives up and stops early.
      Undefined if ``STOP_PATIENCE`` is `None`.
      Initial value: ``STOP_PATIENCE``.
    - ``primal_solver`` (`dict`): ``state_dict`` of the primal problem solver
    - ``dual_solver`` (`dict`): ``state_dict`` of the dual problem solver
    - ``primal_value`` (`float`): Current primal value, i.e., objective function value
    - ``primal_grad_norm`` (`float`): Squared norm of the primal gradient.
    - ``lagrangian_value`` (`float`): Current value of the Lagrangian.
      If the problem is convex, converges to the value of the dual function.
    - ``dual_grad_norm`` (`float`): Squared norm of the dual gradient (constraint slacks).
    - ``duality_gap`` (`float`): Duality gap, i.e., P - D
    - ``rel_duality_gap`` (`float`): Relative duality gap, i.e., (P - D)/P
    - ``constraint_feas`` (`np.array`): Constraint violation of average constraints.
      Non-positive if the solution satisfies the constraint.
    - ``constraint_rel_feas`` (`np.array`): Relative constraint violation of
      average constraints, i.e., slack divided by right-hand side.
      Non-positive if the solution satisfies the constraint.
    - ``constraint_nfeas`` (`np.array`): Proportion of feasible average constraints.
    - ``lambdas_max`` (`float`): Maximum value of dual variables (average constraints).
    - ``pointwise_feas`` (`np.array`): Constraint violation of pointwise constraints.
      Non-positive if the solution satisfies the constraint.
    - ``pointwise_nfeas`` (`np.array`): Proportion of feasible points for each pointwise constraint.
    - ``mus_max`` (`float`): Maximum value of dual variables (pointwise constraints).
    - ``primal_value_log`` (`np.array`): Primal value across iterations.
    - ``lagrangian_value_log`` (`np.array`): Lagrangian value across iterations.
    - ``lambdas_log`` (`np.array`): Dual variables (average constraints) across iterations.
    - ``feas_log`` (`np.array`): Constraint violation (average constraints) across iterations.
    - ``rel_feas_log`` (`np.array`): Relative constraint violation (average constraints) across iterations.
    - ``mus_log`` (`np.array`): Dual variables (pointwise constraints) across iterations.
    - ``nfeas_log`` (`np.array`): Proportion of feasible pointwise constraints across iterations.
    - ``HAS_CONSTRAINTS`` (`bool`): `True` if learning problem has constraints and `False` otherwise.
    - ``N_AVG_CONSTRAINTS`` (`int`): Number of average constraints of learning problem.
    - ``N_PTW_CONSTRAINTS`` (`int`): Number of pointwise constraints of learning problem.

    Except for ``iteration``, states ending in ``_log``, and flags (capitalized states),
    the value of the state in the previous iteration can be accessed by appending ``_prev``
    to the state variable name.

    .. warning:: The values of ``primal_value``, ``lagrangian_value``, ``duality_gap``,
                 ``rel_duality_gap``, ``constraint_feas``, ``constraint_rel_feas``,
                 ``pointwise_feas``, ``pointwise_nfeas`` should be considered as
                 estimates unless ``COMPUTE_TRUE_DGAP`` is `True`. This is particularly
                 an issue when the solver uses batches rather than operating over
                 the full dataset (e.g., SGD). In these cases, the primal and/or
                 dual variables are updated between between batches, so their
                 average is not representative of the current performance. When
                 ``COMPUTE_TRUE_DGAP`` is `True`, the base solver does an extra pass
                 through the dataset in order to re-evaluate these quantities
                 for the current model. This may considerably increase the
                 computation time of each epoch.

    """

    def __init__(self, settings):
        """Primal-dual base solver constructor

        Parameters
        ----------
        settings : `SolverSettings`
            Solver settings.

        """
        self.primal_solver = None
        self.primal_step_size = None
        self.dual_solver = None
        self.dual_step_size = None
        self.state_dict = {}

        self.settings = settings


    def primal_dual_update(self, problem):
        """Primal-dual update

        Parameters
        ----------
        problem : `csl.ConstrainedLearningProblem`
            Constrained learning problem.

        Returns
        -------
        primal_value_est : `float`
            Estimate of the primal value.
        primal_grad_norm_est : `float`
            Estimate of the primal gradient squared norm.
        constraint_slacks_est : `list` [`torch.tensor`, (1, )] or `None`
            Estimate of the value of average constraints or `None` if there was no dual update.
        pointwise_slacks_est : `list` [`torch.tensor`, (N, )] or `None`
            Estimate of the value of pointwise constraints or `None` if there was no dual update.
        dual_grad_norm_est : `float` or `None`
            Estimate of the dual gradient squared norm or `None` if there was no dual update.

        """
        raise NotImplementedError


    def solve(self, problem, **kwargs):
        """Solve constrained learning problem

        Parameters
        ----------
        problem : `csl.ConstrainedLearningProblem`
            Constrained learning problem to solve.
        **kwargs : `dict`, optional
            Temporary settings to override the global solver settings for current run.

        """
        ### Local settings for current run
        self.settings.override(kwargs)

        ### Initializations
        self._initialize(problem)
        self._print_log(self.settings['logger'], header=True)

        ### START OF ITERATIONS ###
        for self.state_dict['iteration'] in range(self.state_dict['iteration'], self.state_dict['iteration'] + self.settings['iterations']):
            ### PRIMAL ###
            primal_value, primal_grad_norm, constraint_slacks, pointwise_slacks, dual_grad_norm = self.primal_dual_update(problem)

            # Update primal step size
            if self.primal_step_size is not None:
                self.primal_step_size.step()

            # Recompute objective on current primal solution
            if self.settings['COMPUTE_TRUE_DGAP']:
                with torch.no_grad():
                    primal_value = problem.objective()

            # Log primal iteration
            self._log_primal(primal_value, primal_grad_norm)


            ### DUAL ###
            if self.state_dict['HAS_CONSTRAINTS']:
                # Update dual step size
                if self.dual_step_size is not None:
                    self.dual_step_size.step()

                # Recompute Lagrangian and slacks on current primal/dual variables solution
                if self.settings['COMPUTE_TRUE_DGAP']:
                    with torch.no_grad():
                        L, _, constraint_slacks, pointwise_slacks = problem.lagrangian()

                        dual_grad_norm = 0
                        for ii, slack in enumerate(constraint_slacks):
                            if problem.lambdas[ii] > 0 or (problem.lambdas[ii] == 0 and slack > 0):
                                dual_grad_norm += slack.item()**2
                        for ii, slack in enumerate(pointwise_slacks):
                            inactive = torch.logical_or(problem.mus[ii] > 0, \
                                                        torch.logical_and(problem.mus[ii] == 0, slack > 0))
                            dual_grad_norm += slack[inactive].norm().item()**2
                else:
                    if constraint_slacks is None and pointwise_slacks is None:
                        L = None
                    else:
                        L = primal_value
                        if constraint_slacks is not None:
                            L += np.sum([lambda_value*slack for lambda_value, slack in zip(problem.lambdas, constraint_slacks)]).item()
                        if pointwise_slacks is not None:
                            L += np.sum([torch.dot(mu_value, slack) for mu_value, slack in zip(problem.mus, pointwise_slacks)]).item()

                # Log dual iteration
                self._log_dual(L, dual_grad_norm, constraint_slacks, problem.rhs, problem.lambdas,
                               pointwise_slacks, problem.pointwise_rhs, problem.mus)

            # Early stopping
            if self._check_stopping_criteria():
                break

            if self.settings['STOP_USER_DEFINED'] and self.settings['STOP_USER_DEFINED'](problem, self.state_dict):
                self.settings['logger'].info('Stopping criterion: user-defined')
                break

            # Print log
            if self._every(self.settings['verbose']):
                self._print_log(self.settings['logger'])
            else:
                self._print_log(self.settings['logger'], level=logging.DEBUG)
        ### END OF ITERATIONS ###


    def plot(self):
        """Trace plots of solver

        If the problem has no constraints, displays a trace plot of the
        primal value estimate (see `PrimalDualBase`).

        If the problem has constraints, displays trace plot of primal value,
        lagrangian value, relative duality gap, dual variables, and
        feasibility (constraint violation of average constraints and
        proportion of feasible pointwise constraints).

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            MATPLOTLIB figure handle.
        axes : `matplotlib.axes.Axes`
            MATPLOTLIB axes handle.

        """
        n_iter = len(self.state_dict['primal_value_log'])
        iter_idx = np.arange(1, n_iter+1)

        ### Create figure
        if not self.state_dict['HAS_CONSTRAINTS']:
            fig, axes = plot.subplots(1,1)

            ### Solver traces
            # Primal value plot
            axes.plot(iter_idx, self.state_dict['primal_value_log'])
            axes.grid()
            axes.autoscale()
        else:
            if (self.state_dict['N_AVG_CONSTRAINTS'] > 0) and (self.state_dict['N_PTW_CONSTRAINTS'] > 0):
                fig, axes = plot.subplots(2, 3, sharex = True)
                # plot.tight_layout()
            else:
                fig, axes = plot.subplots(2, 2, sharex = True)
                # plot.tight_layout()

            ### Solver traces
            P = self.state_dict['primal_value_log']
            D = self.state_dict['lagrangian_value_log']

            # Primal-dual values plot
            axes[0,0].set_title('Duality gap')
            axes[0,0].plot(iter_idx, P, label = 'Primal')
            axes[0,0].plot(iter_idx, D, label = 'Dual')
            axes[0,0].legend()

            # Relative duality gap plot
            axes[1,0].set_title('Relative duality gap')
            axes[1,0].plot(iter_idx, np.abs((P-D)/P))
            axes[1,0].set_yscale('log')

            if self.state_dict['N_AVG_CONSTRAINTS'] > 0:
                # Average constraints slacks
                axes[0,1].set_title('Slacks (average)')

                ax1 = axes[0,1]
                for ii in range(self.state_dict['lambdas_log'].shape[1]):
                    ax1.plot(iter_idx, self.state_dict['feas_log'][:,ii])

                # Dual variables (average)
                axes[1,1].set_title('Dual variables (average)')
                axes[1,1].plot(iter_idx, self.state_dict['lambdas_log'])

            if self.state_dict['N_PTW_CONSTRAINTS'] > 0:
                if self.state_dict['N_AVG_CONSTRAINTS'] > 0:
                    ax1 = axes[0,2]
                    ax2 = axes[1,2]
                else:
                    ax1 = axes[0,1]
                    ax2 = axes[1,1]

                # Pointwise constraints coverage
                ax1.set_title('Feasibility coverage (pointwise)')
                ax1.plot(iter_idx, self.state_dict['nfeas_log'], color='C1')

                ax2.set_title('Dual variables (pointwise)')
                for ii in range(self.state_dict['mus_log'].shape[1]):
                    ax2.fill_between(iter_idx,
                                     self.state_dict['mus_log'][:,ii,1],
                                     self.state_dict['mus_log'][:,ii,2],
                                     alpha = 0.25, color = f'C{ii}')
                    ax2.plot(iter_idx, self.state_dict['mus_log'][:,ii,0],
                             color = f'C{ii}')

            for ax in axes.flat:
                ax.grid()
                ax.autoscale()

        # Show figure
        fig.show()

        return fig, axes


    def reset(self):
        """Reset the constrained learning solver

        Allows a single solver object to be used to solve
        multiple constrained learning problems.

        """
        self.primal_solver = None
        self.primal_step_size = None
        self.dual_solver = None
        self.dual_step_size = None
        self.state_dict = {}


    ###########################################################################
    #### PRIVATE FUNCTIONS                                                 ####
    ###########################################################################

    #### UTILS ####
    def _every(self, period):
        """Test for periodic tasks.

        Checks if ``iteration`` counter is a multiple of ``period``.

        Parameters
        ----------
        period : `int` or `None`
            Period of test. Disabled if 0 or `None` (i.e., returns `False`).

        Returns
        -------
        bool
            `True` if ``iteration`` is multiple of ``period`` or `False` otherwise.

        """
        if period is None or period < 1:
            return False
        else:
            return self.state_dict['iteration'] % period == period-1


    def _update_state(self, **kwargs):
        """Update state variable while keeping previous value.

        Parameters
        ----------
        **kwargs
            ``name=value`` pairs to update in state dictionary

        """
        for key, value in kwargs.items():
            if key in self.state_dict:
                self.state_dict[key+'_prev'] = self.state_dict[key]
            else:
                self.state_dict[key+'_prev'] = value

            if value is not None:
                self.state_dict[key] = value


    #### EARLY STOPPING ####
    def _check_stopping_criteria(self):
        """Check stopping criteria.

        Returns
        -------
        `bool`
            `True` if solver should stop or `False` otherwise.

        See also
        --------
        `PrimalDualBase`

        """
        stopping = False

        # Divergence
        if self._has_diverged('primal_value') or self._has_diverged('lagrangian_value') or \
            self._has_diverged('lambdas_max') or self._has_diverged('mus_max'):
                self.settings['logger'].error(f"Algorithm diverged on iteration {self.state_dict['iteration']}")
                self.settings['logger'].debug(f"Primal = {self.state_dict.get('primal_value', 0)}")
                self.settings['logger'].debug(f"Lagrangian = {self.state_dict.get('lagrangian_value', 0)}")
                self.settings['logger'].debug(f"Dual variable (average) = {self.state_dict.get('lambdas_max', 0)}")
                self.settings['logger'].debug(f"Dual variable (pointwise) = {self.state_dict.get('mus_max', 0)}")
                return True

        if not self.state_dict['HAS_CONSTRAINTS']:
            # Primal absolute optimality
            if self._is_primal_optimal():
                self.settings['logger'].info('Stopping criterion: primal absolute near-optimality')
                stopping = True
        else:
            # Primal absolute optimality and absolute feasibility
            if self._is_primal_optimal() and self._is_absolutely_feasible():
                self.settings['logger'].info('Stopping criterion: primal absolute near-optimality and primal absolute feasibility')
                stopping = True

            # Primal-dual absolute optimality and absolute feasibility
            if self._is_primal_dual_absolutely_optimal() and self._is_absolutely_feasible():
                self.settings['logger'].info('Stopping criterion: primal-dual absolute near-optimality and absolute feasibility')
                stopping = True

            # Primal-dual relative optimality and relative feasibility
            if self._is_primal_dual_relatively_optimal() and self._is_relatively_feasible():
                self.settings['logger'].info('Stopping criterion: primal-dual relative near-optimality and relative feasibility')
                stopping = True

        # No primal or feasibility improvement
        if self.settings['STOP_PATIENCE']:
            if self._is_stalled():
                self._update_state(no_update_iterations = self.state_dict['no_update_iterations'] - 1)
                if self.state_dict['no_update_iterations'] == 0:
                    self.settings['logger'].info("Stopping criterion: no improvement for "
                                                 f"{self.settings['STOP_PATIENCE']} iterations.")
                    stopping = True
            else:
                self._update_state(no_update_iterations = self.settings['STOP_PATIENCE'])

        return stopping


    def _has_diverged(self, state):
        """Test if variable diverged.

        Test against ``STOP_DIVERGENCE`` and `np.nan`.


        Parameters
        ----------
        state : `str`
            State name.

        Returns
        -------
        `bool`
            `True` if diverged or `False` otherwise.

        """
        return self.state_dict.get(state, 0) >= self.settings['STOP_DIVERGENCE'] or \
            np.isnan(self.state_dict.get(state, 0))


    def _is_primal_optimal(self):
        """Test if solution is primal optimal.

        Test if objective value and primal gradient squared norm are small
        (``STOP_PVAL`` and ``STOP_PGRAD``).

        Returns
        -------
        `bool`
            `True` if primal optimal or `False` otherwise.

        """
        if self.settings['STOP_PVAL'] or self.settings['STOP_PGRAD']:
            truth_value = True

            if self.settings['STOP_PVAL']:
                truth_value = truth_value and \
                    (self.state_dict['primal_value'] < self.settings['STOP_PVAL'])

            if self.settings['STOP_PGRAD']:
                truth_value = truth_value and \
                    (self.state_dict['primal_grad_norm'] < self.settings['STOP_PGRAD'])
        else:
            truth_value = False

        return truth_value


    def _is_primal_dual_absolutely_optimal(self):
        """Test if solution is primal-dual absolutely optimal.

        Test if duality gap is small (``STOP_ABS_DGAP``) or
        if gradient squared norms are small (``STOP_PGRAD`` and ``STOP_DGRAD``).

        Returns
        -------
        `bool`
            `True` if primal-dual absolutely optimal or `False` otherwise.

        """
        if self.settings['STOP_ABS_DGAP']:
            truth_value = (self.state_dict['duality_gap'] < self.settings['STOP_ABS_DGAP'])
        elif self.settings['STOP_PGRAD'] and self.settings['STOP_DGRAD']:
            truth_value = (self.state_dict['primal_grad_norm'] < self.settings['STOP_PGRAD']) and \
                (self.state_dict['dual_grad_norm'] < self.settings['STOP_DGRAD'])
        else:
            truth_value = False

        return truth_value


    def _is_primal_dual_relatively_optimal(self):
        """Test if solution is primal-dual relatively optimal.

        Test if relative duality gap is small (``STOP_REL_DGAP``).

        Returns
        -------
        `bool`
            `True` if primal-dual relatively optimal or `False` otherwise.

        """
        if self.settings['STOP_REL_DGAP']:
            return self.state_dict['rel_duality_gap'] < self.settings['STOP_REL_DGAP']


    def _is_absolutely_feasible(self):
        """Test if solution is absolutely feasible.

        Test if absolute constraint violation (average) is small and
        proportion of feasible constraints (pointwise) is large
        (``STOP_ABS_FEAS``, ``STOP_NFEAS``).

        Returns
        -------
        `bool`
            `True` if absolutely feasible or `False` otherwise.

        """
        if self.settings['STOP_ABS_FEAS'] or self.settings['STOP_NFEAS']:
            truth_value = True

            if self.settings['STOP_ABS_FEAS']:
                truth_value = truth_value and \
                    (np.max(self.state_dict.get('constraint_feas', 0)) < self.settings['STOP_ABS_FEAS'])

            if self.settings['STOP_NFEAS']:
                truth_value = truth_value and \
                    (np.max(self.state_dict.get('pointwise_nfeas', 1)) >= self.settings['STOP_NFEAS'])
        else:
            truth_value = False

        return truth_value


    def _is_relatively_feasible(self):
        """Test if solution is relatively feasible.

        Test if relative constraint violation (average) is small and
        proportion of feasible constraints (pointwise) is large
        (``STOP_REL_FEAS``, ``STOP_NFEAS``).

        Returns
        -------
        `bool`
            `True` if relatively feasible or `False` otherwise.

        """
        if self.settings['STOP_REL_FEAS'] or self.settings['STOP_NFEAS']:
            truth_value = True

            if self.settings['STOP_REL_FEAS']:
                truth_value = truth_value and \
                    (np.max(self.state_dict.get('constraint_rel_feas', 0)) < self.settings['STOP_REL_FEAS'])

            if self.settings['STOP_NFEAS']:
                truth_value = truth_value and \
                    (np.max(self.state_dict.get('pointwise_nfeas', 1)) >= self.settings['STOP_NFEAS'])
        else:
            truth_value = False

        return truth_value


    def _is_stalled(self):
        """Test if solver stalled.

        Test if primal value and constraint feasibility has not improved
        since last iteration.

        Returns
        -------
        `bool`
            `True` if stalled or `False` otherwise.

        """
        truth_value = self.state_dict['primal_value'] >= self.state_dict['primal_value_prev']

        if self.state_dict['N_AVG_CONSTRAINTS'] > 0:
            truth_value = truth_value or \
                np.all(np.clip(self.state_dict['constraint_feas'], 0, None) >=
                       np.clip(self.state_dict['constraint_feas_prev'], 0, None))

        if self.state_dict['N_PTW_CONSTRAINTS'] > 0:
            truth_value = truth_value or \
                np.all(self.state_dict['pointwise_nfeas'] <= self.state_dict['pointwise_nfeas_prev'])

        return truth_value


    #### INITIALIZATION ####
    def _initialize(self, problem):
        """Initialize solver.

        - Initialize state dictionary
        - Initialize logs (see `_initialize_logs`)
        - Initialize primal solver and primal step size scheduler.
          Load previous state if it exists.
        - Initialize dual solver and dual step size scheduler.
          Load previous state if it exists.

        """
        ### Initialize state dictionary
        if 'iteration' not in self.state_dict:
            self.state_dict['iteration'] = 0
        else:
            self.state_dict['iteration'] += 1

        self.state_dict['N_AVG_CONSTRAINTS'] = len(problem.constraints)
        self.state_dict['N_PTW_CONSTRAINTS'] = len(problem.pointwise)
        self.state_dict['HAS_CONSTRAINTS'] = \
            (self.state_dict['N_AVG_CONSTRAINTS'] > 0) or (self.state_dict['N_PTW_CONSTRAINTS'] > 0)

        if self.settings['STOP_PATIENCE'] is not None:
            self.state_dict['no_update_iterations'] = self.settings['STOP_PATIENCE']

        ### Initialize logs
        self._initialize_logs()

        ### Initialize primal solver
        if self.primal_solver is None:
            self.primal_solver = self.settings['primal_solver'](problem.model.parameters)

        # Load previous settings, if they exist
        if 'primal_solver' in self.state_dict:
            self.primal_solver.load_state_dict(self.state_dict['primal_solver'])

        # Initialize primal step size scheduler
        if self.settings['lr_p_scheduler'] is not None:
            self.primal_step_size = self.settings['lr_p_scheduler'](self.primal_solver)

        ### Initialize dual solver
        if self.dual_solver is None and (problem.lambdas or problem.mus):
            self.dual_solver = self.settings['dual_solver'](problem.lambdas + problem.mus)
        # Load previous settings, if they exist
        if 'dual_solver' in self.state_dict:
            self.dual_solver.load_state_dict(self.state_dict['dual_solver'])

        # Initialize dual step size scheduler
        if self.settings['lr_d_scheduler'] is not None:
            self.dual_step_size = self.settings['lr_d_scheduler'](self.dual_solver)


    def _initialize_logs(self):
        """Initialize logs.

        Initializes logging variables if they are not defined
        in the state dictionary (``primal_value_log``, ``lagrangian_value_log``,
        ``feas_log``, ``rel_feas_log``, ``lambdas_log``, ``nfeas_log``, ``mus_log``)

        """
        if 'primal_value_log' not in self.state_dict:
            self.state_dict['primal_value_log'] = np.zeros(0)

        if self.state_dict['HAS_CONSTRAINTS'] and 'lagrangian_value_log' not in self.state_dict:
            self.state_dict['lagrangian_value_log'] = np.zeros(0)

        if self.state_dict['N_AVG_CONSTRAINTS'] > 0:
            if 'feas_log' not in self.state_dict:
                self.state_dict['feas_log'] = np.zeros([0, self.state_dict['N_AVG_CONSTRAINTS']])
            if 'rel_feas_log' not in self.state_dict:
                self.state_dict['rel_feas_log'] = np.zeros([0, self.state_dict['N_AVG_CONSTRAINTS']])
            if 'lambdas_log' not in self.state_dict:
                self.state_dict['lambdas_log'] = np.zeros([0, self.state_dict['N_AVG_CONSTRAINTS']])

        if self.state_dict['N_PTW_CONSTRAINTS'] > 0:
            if 'nfeas_log' not in self.state_dict:
                self.state_dict['nfeas_log'] = np.zeros([0, self.state_dict['N_PTW_CONSTRAINTS']])
            if 'mus_log' not in self.state_dict:
                self.state_dict['mus_log'] = np.zeros([0, self.state_dict['N_PTW_CONSTRAINTS'], 3])


    #### LOGGING ####
    def _print_log(self, logger, level=logging.INFO, header=False):
        """Print log to logger.

        Print a formatted table to logger.

        - ``Iteration``: iteration number
        - ``P``: primal value
        - ``PGRAD``: primal gradient squared norm
        - ``DGAP``: duality gap (shown only if problem has constraints)
        - ``RDGAP``: relative duality gap (shown only if problem has constraints)
        - ``DGRAD``: dual gradient squared norm (shown only if problem has constraints)
        - ``FEAS_AVG``: maximum constraint violation of average constraints (shown only if problem has average constraints)
        - ``RFEAS_AVG``: maximum relative constraint violation of average constraints (shown only if problem has average constraints)
        - ``FEAS_PTW``: maximum constraint violation of pointwise constraints (shown only if problem has pointwise constraints)
        - ``NFEAS_PTW``: proportion of feasible pointwise constraints (shown only if problem has pointwise constraints)

        Parameters
        ----------
        logger : `logging`
            Python logging object to handle log.
        level : `int`, optional
            Logging level (see `logging` package for details). The default is logging.INFO.
        header : `bool`, optional
            `True` to print log header or `False` to print log. The default is `False` (print log).

        """
        if self.state_dict['HAS_CONSTRAINTS']:
            message = {'Iteration': f"{self.state_dict.get('iteration', 0) + 1}",
                       'P':         f"{self.state_dict.get('primal_value', 0):.3g}",
                       'DGAP':      f"{self.state_dict.get('duality_gap', 0):.3g}",
                       'RDGAP':     f"{self.state_dict.get('rel_duality_gap', 0):.3g}",
                       'PGRAD':     f"{self.state_dict.get('primal_grad_norm', 0):.3g}",
                       'DGRAD':     f"{self.state_dict.get('dual_grad_norm', 0):.3g}"}

            if self.state_dict['N_AVG_CONSTRAINTS'] > 0:
                message['FEAS_AVG'] = f"{np.max(self.state_dict.get('constraint_feas', 0)):.3g}"
                message['RFEAS_AVG'] = f"{np.max(self.state_dict.get('constraint_rel_feas', 0)):.3g}"

            if self.state_dict['N_PTW_CONSTRAINTS'] > 0:
                message['FEAS_PTW'] = f"{np.max(self.state_dict.get('pointwise_feas', 0)):.3g}"
                message['NFEAS_PTW'] = f"{np.max(self.state_dict.get('pointwise_nfeas', 0)):.3g}"
        else:
            message = {'Iteration': f"{self.state_dict.get('iteration', 0) + 1}",
                       'P':         f"{self.state_dict.get('primal_value', 0):.3g}",
                       'PGRAD':     f"{self.state_dict.get('primal_grad_norm', 0):.3g}"}

        if header:
            logger.log(level, ('{:<9} | '*(len(message)-1) + '{:<9}').format(*message.keys()))
        else:
            logger.log(level, ('{:<9} | '*(len(message)-1) + '{:<9}').format(*message.values()))


    def _log_primal(self, primal_value, primal_grad_norm):
        """Log primal step.

        Parameters
        ----------
        primal_value : `float`
            Saved to ``primal_value`` and ``primal_value_log``.
        primal_grad_norm : `float`
            Saved to ``primal_grad_norm``.

        """
        ### Update state ###
        self.state_dict['primal_solver'] = self.primal_solver.state_dict()
        self._update_state(primal_value = primal_value,
                           primal_grad_norm = primal_grad_norm)

        ### Update trace ###
        self.state_dict['primal_value_log'] = np.append(self.state_dict['primal_value_log'],
                                                        self.state_dict['primal_value'])


    def _log_dual(self, lagrangian_value, dual_grad_norm, constraint_slacks, rhs, lambdas,
                  pointwise_slacks, pointwise_rhs, mus):
        """Log dual step.

        Parameters
        ----------
        lagrangian_value : `float`
            Saved to ``lagrangian_value`` and ``lagrangian_value_log``. Used to compute ``duality_gap``.
        dual_grad_norm : `float`
            Saved to ``dual_grad_norm``.
        constraint_slacks : `list` [`float`]
            Saved to ``constraint_feas``. Used to compute ``constraint_rel_feas`` and ``constraint_nfeas``.
        rhs : `list` [`float`]
            Right-hand side of average constraints. Used  to compute ``constraint_rel_feas``.
        lambdas : `list` [`torch.tensor`, (1, )]
            Saved to ``lambdas_log``.
        pointwise_slacks : `list` [`torch.tensor`, (N, )]
            Saved to ``pointwise_feas``. Used to compute ``pointwise_rel_feas`` and ``pointwise_nfeas``.
        pointwise_rhs : `list` [`torch.tensor`, (N, )]
            Right-hand side of pointwise constraints. Used  to compute ``pointwise_rel_feas``.
        mus : `list` [`torch.tensor`, (N, )]
            Saved to ``mus_log``.

        """
        ### Update state ###
        self.state_dict['dual_solver'] = self.dual_solver.state_dict()

        # Dual value
        self._update_state(lagrangian_value = lagrangian_value,
                           dual_grad_norm = dual_grad_norm)
        self._update_state(duality_gap = np.abs(self.state_dict['primal_value'] - self.state_dict['lagrangian_value']))

        # Duality gap
        if np.abs(self.state_dict['primal_value']) < 1e-6:
            self._update_state(rel_duality_gap = np.float("Inf"))
        else:
            self._update_state(rel_duality_gap = self.state_dict['duality_gap']/self.state_dict['primal_value'])

        # Average constraint feasibility
        if self.state_dict['N_AVG_CONSTRAINTS'] > 0 and constraint_slacks is not None:
            self._update_state(constraint_feas = np.array([slacks.detach().to('cpu') for slacks in constraint_slacks], ndmin=2))
            self._update_state(constraint_rel_feas = np.array([(slack.detach()/c).to('cpu') if (c != 0) else float("Inf") \
                                                               for slack, c in zip(constraint_slacks, rhs)], ndmin=2))
            self._update_state(constraint_nfeas = np.mean(self.state_dict.get('constraint_feas') <= 0, axis=1))

        # Pointwise constraint feasibility
        if self.state_dict['N_PTW_CONSTRAINTS'] > 0 and pointwise_slacks is not None:
            self._update_state(pointwise_feas = np.concatenate([slacks.detach().to('cpu') for slacks in pointwise_slacks]))
            self._update_state(pointwise_rel_feas = np.concatenate([(slack.detach()/c).to('cpu') if torch.all(c != 0) else float("Inf") \
                                                                    for slack, c in zip(pointwise_slacks, pointwise_rhs)]))
            self._update_state(pointwise_nfeas = np.array([np.mean(value.detach().to('cpu').numpy() <= 0) for value in pointwise_slacks], ndmin=2))


        ### Update traces ###
        # Dual value
        self.state_dict['lagrangian_value_log'] = np.append(self.state_dict['lagrangian_value_log'],
                                                            self.state_dict['lagrangian_value'])

        # Average constraint feasibility
        if self.state_dict['N_AVG_CONSTRAINTS'] > 0:
            self.state_dict['feas_log'] = np.append(self.state_dict['feas_log'],
                                                    self.state_dict.get('constraint_feas', 0),
                                                    axis=0)
            self.state_dict['rel_feas_log'] = np.append(self.state_dict['rel_feas_log'],
                                                        self.state_dict.get('constraint_rel_feas', 0),
                                                        axis=0)
            self.state_dict['lambdas_log'] = np.append(self.state_dict['lambdas_log'],
                                                       np.array([lambda_value.to('cpu') for lambda_value in lambdas], ndmin = 2),
                                                       axis=0)
            self._update_state(lambdas_max = np.max(self.state_dict['lambdas_log'][-1,:]))

        # Pointwise constraint feasibility
        if self.state_dict['N_PTW_CONSTRAINTS'] > 0:
            self.state_dict['nfeas_log'] = np.append(self.state_dict['nfeas_log'],
                                                     self.state_dict['pointwise_nfeas'],
                                                     axis=0)
            self.state_dict['mus_log'] = np.append(self.state_dict['mus_log'],
                                                   np.array([[mu_value.to('cpu').numpy().mean(),
                                                              mu_value.to('cpu').numpy().min(),
                                                              mu_value.to('cpu').numpy().max()] \
                                                             for mu_value in mus], ndmin=3),
                                                   axis=0)
            self._update_state(mus_max = np.max(self.state_dict['mus_log'][-1,:,2]))
