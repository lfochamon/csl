#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""Fairness application

Adult dataset with (asymmetrical) demographic parity constraint

"""

    import torch
    import torch.nn.functional as F
    import torchvision

    import matplotlib.pyplot as plot

    import functools

    import csl
    import csl.datasets

    import os
    import logging


    ####################################
    # DATA                             #
    ####################################
    # Preprocessing
    preprocess = torchvision.transforms.Compose([
        csl.datasets.utils.Drop(['fnlwgt', 'educational-num', 'relationship', 'capital-gain', 'capital-loss']),
        csl.datasets.utils.Recode('education', {'<= K-12': ['Preschool', '1st-4th', '5th-6th', '7th-8th',
                                          '9th', '10th', '11th', '12th']}),
        csl.datasets.utils.Recode('race', {'Other': ['Other', 'Amer-Indian-Eskimo']}),
        csl.datasets.utils.Recode('marital-status', {'Married': ['Married-civ-spouse', 'Married-AF-spouse',
                                              'Married-spouse-absent'],
                                  'Divorced/separated': ['Divorced', 'Separated']}),
        csl.datasets.utils.Recode('native-country', {'South/Central America': ['Columbia', 'Cuba', 'Guatemala',
                                                            'Haiti', 'Ecuador', 'El-Salvador',
                                                            'Dominican-Republic', 'Honduras',
                                                            'Jamaica', 'Nicaragua', 'Peru',
                                                            'Trinadad&Tobago'],
                                  'Europe': ['England', 'France', 'Germany', 'Greece',
                                              'Holand-Netherlands', 'Hungary', 'Italy',
                                              'Ireland', 'Portugal', 'Scotland', 'Poland',
                                              'Yugoslavia'],
                                  'Southeast Asia': ['Cambodia', 'Laos', 'Philippines',
                                                      'Thailand', 'Vietnam'],
                                  'Chinas': ['China', 'Hong', 'Taiwan'],
                                  'USA': ['United-States', 'Outlying-US(Guam-USVI-etc)',
                                          'Puerto-Rico']}),
        csl.datasets.utils.QuantileBinning('age', 6),
        csl.datasets.utils.Binning('hours-per-week', bins = [0,40,100]),
        csl.datasets.utils.Dummify(csl.datasets.Adult.categorical + ['age', 'hours-per-week'])
        ])

    # Load Adult data
    trainset = csl.datasets.Adult(root = 'data', train = True, target_name = 'income', preprocess = preprocess,
                                  transform = csl.datasets.utils.ToTensor(dtype = torch.float),
                                  target_transform = csl.datasets.utils.ToTensor(dtype = torch.long))

    testset = csl.datasets.Adult(root = 'data', train = False, target_name = 'income', preprocess = preprocess,
                                 transform = csl.datasets.utils.ToTensor(dtype = torch.float),
                                 target_transform = csl.datasets.utils.ToTensor(dtype = torch.long))

    # Gender column index
    fullset = csl.datasets.Adult(root = 'data', train = False, target_name = 'income', preprocess = preprocess)
    gender_idx = [idx for idx, name in enumerate(fullset[0][0].columns) if name.startswith('gender')]


    ####################################
    # MODEL                            #
    ####################################
    class Logistic:
        def __init__(self, n_features):
            self.parameters = [torch.zeros(1, dtype = torch.float, requires_grad = True),
                               torch.zeros([n_features,1], dtype = torch.float, requires_grad = True)]

        def __call__(self, x):
            if len(x.shape) == 1:
                x = x.unsqueeze(1)

            yhat = self.logit(torch.mm(x, self.parameters[1]) + self.parameters[0])

            return torch.cat((1-yhat, yhat), dim=1)

        def predict(self, x):
            _, predicted = torch.max(self(x), 1)
            return predicted

        @staticmethod
        def logit(x):
            return 1/(1 + torch.exp(-x))


    ####################################
    # PROBLEM                          #
    ####################################
    class fairClassification(csl.ConstrainedLearningProblem):
        def __init__(self, rhs = None):
            self.model = Logistic(trainset[0][0].shape[0])
            self.data = trainset
            self.obj_function = self.loss

            if rhs is not None:
                # Gender
                self.constraints = [self.DemographicParity(self, gender_idx, 0),
                                    self.DemographicParity(self, gender_idx, 1)]
                self.rhs = [rhs, rhs]

            super().__init__()


        def loss(self, batch_idx):
            # Evaluate objective
            x, y = self.data[batch_idx]
            yhat = self.model(x)

            return F.cross_entropy(yhat, y) + 1e-3*(self.model.parameters[0]**2 + self.model.parameters[1].norm()**2)


        class DemographicParity:
            def __init__(self, problem, protected_idx, protected_value):
                self.problem = problem
                self.protected_idx = protected_idx
                self.protected_value = protected_value or (1,)*len(protected_idx)

            def __call__(self, batch_idx, primal):
                x, y = self.problem.data[batch_idx]

                group_idx = (x[:, self.protected_idx] == self.protected_value)

                if primal:
                    yhat = self.problem.model(x)
                    pop_indicator = torch.sigmoid(8*(yhat[:,1] - 0.5))
                    group_indicator = torch.sigmoid(8*(yhat[group_idx,1] - 0.5))
                else:
                    yhat = self.problem.model.predict(x)
                    pop_indicator = yhat.float()
                    group_indicator = yhat[group_idx].float()

                return -(group_indicator.mean() - pop_indicator.mean())

    problems = {
      'unconstrained': fairClassification(trainset),
      'constrained': fairClassification(trainset, rhs = 0.01),
      }


    #%% ################################
    # TRAINING                         #
    ####################################
    solver_settings = {'iterations': 700,
                       'batch_size': None,
                       'primal_solver': torch.optim.Adam,
                       'lr_p0': 0.2,
                       'dual_solver': torch.optim.Adam,
                       'lr_d0': 0.001,
                       }
    solver = csl.PrimalDual(solver_settings)

    solutions = {}
    for key, problem in problems.items():
        solver.reset()
        solver.solve(problem)
        solver.plot()

        solution[key] = {'model': problem.model,
                         'lambdas': problem.lambdas,
                         'solver_state': solver.state_dict}


     ####################################
     # TESTING                          #
     ####################################
     def accuracy(pred, y):
         correct = (pred == y).sum().item()
         return correct/pred.shape[0]

     def disparity(x, model, protected_idx, protected_value):
         pred = model.predict(x)

         pop_prev = pred.float().mean().item()

         group_idx = fullset[:][0].iloc[:,protected_idx] == protected_value)

         group_prev = pred[group_idx].float().mean().item()

         disparity_value = group_prev - pop_prev
         rel_disparity_value = disparity_value/pop_prev

         return disparity_value, rel_disparity_value

    for key, solution in solutions.items():
        print(f'Model: {key}')
        with torch.no_grad():
            x_test, y_test = problem['testset'][:]
            yhat = solution[key]['model'].predict(x_test)

            acc_test = accuracy(yhat, y_test)

            disparity_f, rel_disparity_f = disparity(x_test, solution[key]['model'], gender_idx, 0)
            disparity_m, rel_disparity_m = disparity(x_test, solution[key]['model'], gender_idx, 1)

            print(f'Test accuracy: {100*acc_test:.2f}')
            print(f'Predicted population prevalence: {100*yhat.float().mean().item():.2f}')
            print(f'Female disparity: {100*disparity_f:.2f} | {100*rel_disparity_f:.2f}')
            print(f'Male disparity: {100*disparity_m:.2f} | {100*rel_disparity_m:.2f}')
