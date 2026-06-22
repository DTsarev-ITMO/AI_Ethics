import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.widgets import Slider


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class Plotter3D:
    def __init__(self, n: int):
        self.n = n
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)

        self.init_elev = 35
        self.init_azim = 70
        self.init_roll = 0

    def plot(self, solution0, solution1):
        ax = self.ax
        max_hub_id = 6  # Внутри метода, если это константа

        # Рисуем фоновую сферу
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="b", alpha=0.1, linewidths=0.5)

        # Оси координат
        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
        ax.add_artist(Arrow3D([0, 1.5], [0, 0], [0, 0], color="r", **arrow_prop_dict))
        ax.add_artist(Arrow3D([0, 0], [0, 1.5], [0, 0], color="g", **arrow_prop_dict))
        ax.add_artist(Arrow3D([0, 0], [0, 0], [0, 1.5], color="b", **arrow_prop_dict))

        ax.view_init(elev=self.init_elev, azim=self.init_azim, roll=self.init_roll)

        # Ползунки
        ax_color = 'lightgoldenrodyellow'
        ax_elev = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=ax_color)
        ax_azim = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=ax_color)
        ax_roll = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=ax_color)

        self.slider_elev = Slider(ax_elev, 'Elev', -90, 90, valinit=self.init_elev)
        self.slider_azim = Slider(ax_azim, 'Azim', -180, 180, valinit=self.init_azim)
        self.slider_roll = Slider(ax_roll, 'Roll', -180, 180, valinit=self.init_roll)

        def update(val):
            ax.view_init(elev=self.slider_elev.val,
                         azim=self.slider_azim.val,
                         roll=self.slider_roll.val)
            self.fig.canvas.draw_idle()

        self.slider_elev.on_changed(update)
        self.slider_azim.on_changed(update)
        self.slider_roll.on_changed(update)

        for sol, color in [(solution1, "b"), (solution1, "r")]:
            idx = 0 if color == "b" else -1
            r = sol['R'][max_hub_id][idx]
            xs = r * np.cos(u) * np.sin(v)
            ys = r * np.sin(u) * np.sin(v)
            zs = r * np.cos(v)
            ax.plot_wireframe(xs, ys, zs, color=color, linewidths=0.5, alpha=0.22)

        for ii in range(self.n):
            ax.plot3D(solution0['px'][ii], solution0['py'][ii], solution0['sigmaz'][ii], color="g", linewidth=0.3)
            ax.plot3D(solution1['px'][ii], solution1['py'][ii], solution1['sigmaz'][ii], color="r", linewidth=0.3)

        ax.set_axis_off()