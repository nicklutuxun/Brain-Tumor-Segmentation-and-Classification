import numpy as np
import torch
from toch.optim import Optimizer

"""
Jaya optimizer

Parameters:
Name			Type			Description
PS 				int 			population size
NDV				int 			number of design variables
TER_COD			dict			termination condition

Note that default TER_COD = {"max_iter": 10000}

Note that can ignore above, since Jaya extends torch.optim.Optimizer

To use:
from JayaOptimizer import Jaya
...
run optimization(init, Jaya, n_iter, lr)
"""
class Jaya(Optimizer):
	def __init__(self, parameters, lr = 1e-5):
		defaults = {"lr": lr}
		super().__init__(parameters, defaults)

	def step(self, closure = None):
		loss = None

		if closure is not none:
			loss = closure()

		best = np.array(self.state.loc[self.state["colName"].idxmin][0: dim])
		worst = np.array(self.state.loc[obj["colName"].idxmax][0: dim])

		X_prime = []

		for i in range(len(self.state)):
			X = np.array(self.state.loc[i][0: dim])
			r1 = np.random.random(dim)
			r2 = np.random.random(dim)

			X_prime.append(X + r1 * (best - abs(X)) - r2 * (worst - abs(X)))

		return max(X_prime, X)