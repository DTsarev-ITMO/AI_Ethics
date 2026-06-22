

def equations(t, Y, N, A, JR, JI):
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
            F1i += A[i, j] * (JR * px[j] + JI * py[j]) * (_c[i] * _d[j] + _c[j] * _d[i])
            F2i += A[i, j] * (JR * py[j] - JI * px[j]) * (_c[i] * _c[j] - _d[i] * _d[j])
            F3i += A[i, j] * (JR * px[j] + JI * py[j]) * (_d[i] * _d[j] - _c[i] * _c[j])
            F4i += A[i, j] * (JR * py[j] - JI * px[j]) * (_c[i] * _d[j] + _c[j] * _d[i])
        F1.append(F1i)
        F2.append(F2i)
        F3.append(F3i)
        F4.append(F4i)
    dpx = - Gamma * px + (_c * px - _d * py) * sigmaz + sigmaz / g ** 2 * (np.asarray(F1) + np.asarray(F2))
    dpy = - Gamma * py + (_c * py + _d * px) * sigmaz + sigmaz / g ** 2 * (np.asarray(F3) + np.asarray(F4))
    dsigmaz = gamma * (sigma0 - sigmaz) - _c * (px ** 2 + py ** 2)
    return np.concatenate((dpx, dpy, dsigmaz))