import networkx as nx
# import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
# import random
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
from scipy.integrate import solve_ivp
import time
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

edge_list_file = "../N60_y2.3.edgelist"
# edge_list_file = "../N500_y2.3.edgelist"
G = nx.read_edgelist(edge_list_file)
A = nx.adjacency_matrix(G)
N = len(G)
k = np.sum(A, axis=1)

max_hub_id = 6
hub_ids = 6, 4, 17, 7, 5, 11
mins_ids = 0, 13, 19, 26, 27, 28, 29, 30, 32, 33, 34, 38, 39, 41, 43, 44, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59

tmax = 100

# Параметры системы

g = 0.5
Delta0 = np.asarray([0 for i in range(N)])
Delta1 = np.asarray([0.5 for i in range(N)])

gamma = 0
Gamma = 0
sigma0 = 0

# JR = 0
# JI = 0

# JR = 0.01
# JI = 0

# JR = 0
# JI = - 0.01

JR = 0
JI = 0.01

# Начальные условия
Jtr0 = 1e-5
sigmaz0 = np.ones(N) * np.sqrt(1 - Jtr0 ** 2)
px0 = np.ones(N) * np.sqrt(Jtr0 ** 2 / 2)
py0 = np.ones(N) * np.sqrt(Jtr0 ** 2 / 2)

Y0 = np.concatenate((px0, py0, sigmaz0))
t_span = (0, tmax)

def _get_components(_Y0, _t_span, _Delta):
  print('Processing script.')
  start_time = time.perf_counter()
  _c = g ** 2 / (1 + _Delta ** 2)
  _d = _Delta * g ** 2 / (1 + _Delta ** 2)
  def system(t, Y):
    # print('.')
    px = Y[:N]
    py = Y[N:2 * N]
    sigmaz = Y[2 * N:]
    F1 = []
    F2 = []
    F3 = []
    F4 = []
    for i in range(N):
      F1i = 0
      F2i = 0
      F3i = 0
      F4i = 0
      for j in range(N):
        F1i += A[i,j] * (JR * px[j] + JI * py[j]) * (_c[i] * _d[j] + _c[j] * _d[i])
        F2i += A[i,j] * (JR * py[j] - JI * px[j]) * (_c[i] * _c[j] - _d[i] * _d[j])
        F3i += A[i,j] * (JR * px[j] + JI * py[j]) * (_d[i] * _d[j] - _c[i] * _c[j])
        F4i += A[i,j] * (JR * py[j] - JI * px[j]) * (_c[i] * _d[j] + _c[j] * _d[i])
      F1.append(F1i)
      F2.append(F2i)
      F3.append(F3i)
      F4.append(F4i)
    dpx = - Gamma * px + (_c * px - _d * py) * sigmaz + sigmaz / g ** 2 * (np.asarray(F1) + np.asarray(F2))
    dpy = - Gamma * py + (_c * py + _d * px) * sigmaz + sigmaz / g ** 2 * (np.asarray(F3) + np.asarray(F4))
    dsigmaz = gamma * (sigma0 - sigmaz) - _c * (px ** 2 + py ** 2)
    return np.concatenate((dpx, dpy, dsigmaz))
  sol = solve_ivp(system, _t_span, _Y0, t_eval = np.linspace(_t_span[0], _t_span[1], 1000), method='RK45')
  px = sol.y[:N]
  py = sol.y[N:2 * N]
  sigmaz = sol.y[2 * N:]
  _Jtr = [np.sqrt(px[i] ** 2 + py[i] ** 2) for i in range(N)]
  _R   = [np.sqrt(sigmaz[i] ** 2 + _Jtr[i] ** 2) for i in range(N)]

  end_time = time.perf_counter()
  execution_time = end_time - start_time

  print(f"Script finished for {execution_time:.4f} seconds.")
  return sol.t, px, py, sigmaz, _Jtr, _R

def _draw_2D(_solution0, _solution1, _tmax, _draw_hubs):
  fig1, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

  for ii in range(N):
    ax1.plot(_solution0[0], _solution0[4][ii], 'g-', linewidth = '0.3')
    ax1.plot(_solution0[0], _solution0[3][ii], 'b:', linewidth = '0.6')
    ax1.plot(_solution1[0], _solution1[4][ii], 'r-', linewidth = '0.3')
    ax1.plot(_solution1[0], _solution1[3][ii], 'm:', linewidth = '0.6')

  ax1.plot(_solution0[0], _solution0[4][max_hub_id], 'g-', linewidth = '0.3', label = r'$\sigma_i^\perp$, $\Delta_i = 0.0$')
  ax1.plot(_solution0[0], _solution0[3][max_hub_id], 'b:', linewidth = '0.6', label = r'$\sigma_i^z$,  $\Delta_i = 0.0$')
  ax1.plot(_solution1[0], _solution1[4][max_hub_id], 'r-', linewidth = '0.3', label = r'$\sigma_i^\perp$, $\Delta_i = 0.5$')
  ax1.plot(_solution1[0], _solution1[3][max_hub_id], 'm:', linewidth = '0.6', label = r'$\sigma_i^z$,  $\Delta_i = 0.5$')

  if _draw_hubs:
    ax1.plot(_solution1[0], _solution1[4][max_hub_id], 'k-', linewidth = '1.5')
    ax1.plot(_solution1[0], _solution1[3][max_hub_id], 'k:', linewidth = '3.0')

  plt.rc('font', weight='normal', size = 20)
  plt.rc('xtick', labelsize=20)
  plt.rc('ytick', labelsize=20)
  ax1.set_ylabel(r'$\sigma_i^\perp$, $\sigma_i^z$')
  ax1.set_xlabel(r'τ')

  plt.xlim(0, _tmax)
  plt.ylim(-1.3, 1.3)
  plt.legend(fontsize = 20, loc = 'lower left')

  print('J = ', str(JR + 1j * JI))
  plt.show()




class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

if __name__ == '__main__':
    # print('N = ', N)
    # print('k_min = ', np.min(k))
    # print('k_max = ', np.max(k))
    # print('<k> = ', sum(k) / N)
    # print('zeta = ', np.sum(k**2)/np.mean(k)/N)

    solution0 = _get_components(Y0, t_span, Delta0)
    solution1 = _get_components(Y0, t_span, Delta1)

    # _draw_2D(_solution0 = solution0, _solution1 = solution1, _tmax = 100, _draw_hubs = True)

    # Построить 3D рисунок для всех узлов

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))

    # Оси

    r = solution1[5][max_hub_id][-1]

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
    r = solution1[5][max_hub_id][0]
    u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    surf = ax.plot_wireframe(x, y, z, rcount=10000, color="b", linewidths=0.5, alpha=0.05)

    # Конечная сфера для Delta = 0.5 и хаба
    r = solution1[5][max_hub_id][-1]
    u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    surf = ax.plot_wireframe(x, y, z, rcount=10000, color="r", linewidths=0.5, alpha=0.05)

    # Результаты расчета для всех узлов
    for ii in range(N):
      ax.plot3D(solution0[1][ii], solution0[2][ii], solution0[3][ii], color="g", linewidth='0.3')
      ax.plot3D(solution1[1][ii], solution1[2][ii], solution1[3][ii], color="r", linewidth='0.3')

    # Хаб
    ax.plot3D(solution1[1][max_hub_id], solution1[2][max_hub_id], solution1[3][max_hub_id], color="k", linewidth='1',
              label=r'$\Delta_i = 0.5$')
    # ax.view_init(elev=35, azim=70, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок
    ax.view_init(elev=35, azim=70, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок

    # ax.view_init(elev=25, azim=70, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок

    # ax.view_init(elev=30, azim=-60, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок
    # ax.view_init(elev=90, azim=0, roll=0, vertical_axis='z', share=False)  # Здесь можно вращать рисунок

    ax.set_xlabel(r'$p_i^x$')
    ax.set_ylabel(r'$p_i^y$')
    ax.set_zlabel(r'$\sigma_i^z$')

    ax.set_axis_off()

    # plt.legend()

    print('J = ', str(JR + 1j * JI))
    plt.show()



# max_hub_id = 6
# hub_ids = 6, 4, 17, 7, 5, 11
# mins_ids = 0, 13, 19, 26, 27, 28, 29, 30, 32, 33, 34, 38, 39, 41, 43, 44, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59