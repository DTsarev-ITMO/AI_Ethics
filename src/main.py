import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from src.QuantumSystemSimulator import QuantumSystemSimulator
from src.Plotter2D import Plotter2D
from src.Plotter3D import Plotter3D

edge_list_file = "../N60_y2.3.edgelist"
G = nx.read_edgelist(edge_list_file)
A = nx.adjacency_matrix(G)
N = len(G)
k = np.sum(A, axis=1)

# ______Параметры системы______
g = 0.5
# ______Без потерь______
# gamma = 0
# Gamma = 0
# sigma0 = 0
#
# tmax = 100
# ______С потерями 1______
# gamma = 0.5
# Gamma = 0.1
# sigma0 = 0.99

# tmax = 400
# ______С потерями 2______
gamma = 0.05
Gamma = 0.1
sigma0 = 0.99

tmax = 400
# ________________________

Jtr0 = 1e-5
sigmaz0 = np.ones(N) * np.sqrt(1 - Jtr0 ** 2)
px0 = np.ones(N) * np.sqrt(Jtr0 ** 2 / 2)
py0 = np.ones(N) * np.sqrt(Jtr0 ** 2 / 2)

Y0 = np.concatenate((px0, py0, sigmaz0))
t_span = (0, tmax)

# JR = 0
# JI = 0

# JR = 0.01
# JI = 0

# JR = 0
# JI = - 0.01

JR = 0
JI = 0.01

if __name__ == '__main__':
  simulator = QuantumSystemSimulator(N, A, g, Gamma, gamma, sigma0, JR, JI)
  solution0 = simulator.solve(Y0, t_span, delta=0)
  solution1 = simulator.solve(Y0, t_span, delta=0.5)


  # plotter = Plotter2D(N)
  plotter = Plotter3D(N)

  plotter.plot(solution0, solution1)

  plt.show()