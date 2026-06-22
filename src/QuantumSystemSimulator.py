import numpy as np
import time
from scipy.integrate import solve_ivp


class QuantumSystemSimulator:
    def __init__(self, N, A, g, Gamma, gamma, sigma0, JR, JI):
        self.N = N
        self.A = A
        self.g = g
        self.Gamma = Gamma
        self.gamma = gamma
        self.sigma0 = sigma0
        self.JR = JR
        self.JI = JI

    def _compute_derivatives(self, t, Y, c, d):
        N = self.N
        px = Y[:N]
        py = Y[N:2 * N]
        sigmaz = Y[2 * N:]

        px_term = self.JR * px + self.JI * py
        py_term = self.JR * py - self.JI * px

        Fx = c * (self.A @ (px_term * d)) + d * (self.A @ (px_term * c)) + c * (self.A @ (py_term * c)) - d * (self.A @ (py_term * d))
        Fy = d * (self.A @ (px_term * d)) - c * (self.A @ (px_term * c)) + c * (self.A @ (py_term * d)) + d * (self.A @ (py_term * c))

        dpx = -self.Gamma * px + (c * px - d * py) * sigmaz + (sigmaz / self.g ** 2) * Fx
        dpy = -self.Gamma * py + (c * py + d * px) * sigmaz + (sigmaz / self.g ** 2) * Fy
        dsigmaz = self.gamma * (self.sigma0 - sigmaz) - c * (px ** 2 + py ** 2)

        return np.concatenate((dpx, dpy, dsigmaz))

    def solve(self, Y0, t_span, delta, num_points=1000):
        print(f"Processing script for delta={delta}...")
        if type(delta) == int:
            delta = np.asarray([delta for _ in range(self.N)])
        start_time = time.perf_counter()

        c = self.g ** 2 / (1 + delta ** 2)
        d = delta * self.g ** 2 / (1 + delta ** 2)

        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        sol = solve_ivp(
            fun=self._compute_derivatives,
            t_span=t_span,
            y0=Y0,
            args=(c, d),
            method='RK45',
            t_eval=t_eval
        )

        px = sol.y[:self.N]
        py = sol.y[self.N:2 * self.N]
        sigmaz = sol.y[2 * self.N:]

        j_tr = np.sqrt(px ** 2 + py ** 2)
        r = np.sqrt(sigmaz ** 2 + j_tr ** 2)

        execution_time = time.perf_counter() - start_time
        print(f"Script finished in {execution_time:.4f} seconds.")

        return {
            't': sol.t,
            'px': px,
            'py': py,
            'sigmaz': sigmaz,
            'Jtr': j_tr,
            'R': r
        }