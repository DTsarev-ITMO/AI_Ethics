import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

max_hub_id = 6

class Plotter2D:
    def __init__(self, n: int):
        self.n = n

        plt.rc('font', weight='normal', size=20)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)

    def plot(self, solution):
        fig1, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

        for ii in range(self.n):
            ax1.plot(solution['t'], solution['px'][ii], 'r-', linewidth='0.5')
            ax1.plot(solution['t'], solution['py'][ii], 'g-', linewidth='0.5')
            ax1.plot(solution['t'], solution['sigmaz'][ii], 'b-', linewidth='0.5')

        ax1.set_ylabel(r'$p_x$, $p_y$, $\sigma_i^z$')
        ax1.set_xlabel(r'τ')