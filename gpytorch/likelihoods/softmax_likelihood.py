import torch
import gpytorch
from gpytorch.random_variables import GaussianRandomVariable
from .likelihood import Likelihood
import numpy as np

class SoftmaxLikelihood(Likelihood):
    
    def p_y_given_f_A(self, f, A, y):
        f_vec = f.mean()
        numerator = np.exp(A.dot(f_vec.transpose()).dot(y)
        
        denominator = 0
        for i in range(len(f_vec)):
            e_c = np.zeros(len(f_vec))
            e_c[i] = 1
            denominator +=  np.exp(A.dot(f_vec.transpose()).dot(e_c))
        return numerator/denominator

    def monte_carlo_component(self, A, y, p_f_x):
        draw = p_f_x.sample()
        return self.p_y_given_f(draw, A, y)

    def compute_monte_carlo(self, A, y, p_f_x, J):
        monte_carlo_sum = 0
        for _ in range(J):
            monte_carlo_sum += self.monte_carlo_component(A, y, p_f_x)
        return monte_carlo_sum / J


    def forward(self, A, y, p_f_x):
        assert(isinstance(p_f_x, GaussianRandomVariable))
        J = 100
        return self.compute_monte_carlo(A, y, p_f_x, J)

   
