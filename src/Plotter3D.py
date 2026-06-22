import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.integrate import solve_ivp
import time
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

max_hub_id = 6

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

class Plotter3D:
    def __init__(self, n: int):
        self.n = n

    def plot(self, solution0, solution1):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))

        r = max(solution1['R'][max_hub_id][0], solution1['R'][max_hub_id][-1])

        arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='gray', shrinkA=0, shrinkB=0)
        X = Arrow3D([0, r + 0.2], [0, 0], [0, 0], **arrow_prop_dict)
        Y = Arrow3D([0, 0], [0, r + 0.7], [0, 0], **arrow_prop_dict)
        Z = Arrow3D([0, 0], [0, 0], [0, r + 0.4], **arrow_prop_dict)

        ax.add_artist(X)
        ax.add_artist(Y)
        ax.add_artist(Z)

        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='dimgray', shrinkA=0, shrinkB=0)
        X = Arrow3D([r, r + 0.2], [0, 0], [0, 0], **arrow_prop_dict)
        Y = Arrow3D([0, 0], [r, r + 0.7], [0, 0], **arrow_prop_dict)
        Z = Arrow3D([0, 0], [0, 0], [r, r + 0.4], **arrow_prop_dict)
        ax.add_artist(X)
        ax.add_artist(Y)
        ax.add_artist(Z)

        # Начальная сфера для Delta = 0.5 и хаба
        r = solution1['R'][max_hub_id][0]
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        surf = ax.plot_wireframe(x, y, z, rcount=10000, color="b", linewidths=0.5, alpha=0.25)

        # Конечная сфера для Delta = 0.5 и хаба
        r = solution1['R'][max_hub_id][-1]
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        surf = ax.plot_wireframe(x, y, z, rcount=10000, color="r", linewidths=0.5, alpha=0.25)

        # Результаты расчета для всех узлов
        for ii in range(self.n):
            ax.plot3D(solution0['px'][ii], solution0['py'][ii], solution0['sigmaz'][ii], color="g", linewidth='0.3')
            ax.plot3D(solution1['px'][ii], solution1['py'][ii], solution1['sigmaz'][ii], color="r", linewidth='0.3')

        ax.view_init(elev=35, azim=70, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок

        # ax.view_init(elev=25, azim=70, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок

        # ax.view_init(elev=30, azim=-60, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок
        # ax.view_init(elev=90, azim=0, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок

        ax.set_xlabel(r'$p_i^x$')
        ax.set_ylabel(r'$p_i^y$')
        ax.set_zlabel(r'$\sigma_i^z$')

        ax.set_axis_off()
