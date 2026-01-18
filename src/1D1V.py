import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ----------------------------
# Parámetros del experimento (según 4.2)
# ----------------------------
nx = 200
nv = 200
Lx = 4.0 * np.pi          # dominio en x: [0, 4π]
vmax = 6.0                # dominio en v: [-6, 6]

dx = Lx / nx
dv = 2.0 * vmax / nv

theta = 0.5
alpha = 0.01

dt = 0.05                 # estable con semi-Lagrangiano
tmax = 30.0

# ----------------------------
# Mallas
# ----------------------------
x = np.linspace(0.0, Lx, nx, endpoint=False)
v = np.linspace(-vmax, vmax, nv, endpoint=False)

# FFT en x
k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
k[0] = 1.0  # evitar división por cero

# ----------------------------
# Inicialización f(x,v,0) (ecuación 4.2)
# f(x,v,0) = 1/sqrt(2π) * v^2 * exp(-v^2/2) * (1 + α cos(θ x))
# ----------------------------
X, V = np.meshgrid(x, v, indexing='ij')
f = (1.0 / np.sqrt(2.0 * np.pi)) * (V**2) * np.exp(-0.5 * V**2) * (1.0 + alpha * np.cos(theta * X))

# ----------------------------
# Rutinas no-jit: densidad y Poisson
# ----------------------------
def rho_from_f(f):
    return np.sum(f, axis=1) * dv

def solve_poisson(rho):
    rho0 = np.mean(rho)
    rh = np.fft.fft(rho - rho0)
    Eh = 1j * rh / k
    Eh[0] = 0.0
    return np.real(np.fft.ifft(Eh))

# ----------------------------
# Núcleos numba
# ----------------------------
@njit(parallel=True)
def interp_x_periodic(f, x_new, Lx, dx):
    nx, nv = f.shape
    out = np.zeros_like(f)
    for j in prange(nv):
        x_mod = x_new[:, j] % Lx
        xi = x_mod / dx
        i0 = np.floor(xi).astype(np.int64)
        i1 = (i0 + 1) % nx
        w = xi - i0
        for i in range(nx):
            out[i, j] = (1.0 - w[i]) * f[i0[i], j] + w[i] * f[i1[i], j]
    return out

@njit
def interp_v_line(frow, v_new, vmax, dv):
    nv = frow.shape[0]
    out = np.empty_like(v_new)
    for j in range(v_new.shape[0]):
        vv = v_new[j]
        if vv < -vmax + 1e-12:
            vv = -vmax + 1e-12
        elif vv > vmax - 1e-12:
            vv = vmax - 1e-12
        eta = (vv + vmax) / dv
        j0 = int(np.floor(eta))
        if j0 < 0:
            j0 = 0
        if j0 > nv - 2:
            j0 = nv - 2
        j1 = j0 + 1
        w = eta - j0
        out[j] = (1.0 - w) * frow[j0] + w * frow[j1]
    return out

@njit(parallel=True)
def advect_v_numba(f, E, v, vmax, dv, dt_half):
    nx, nv = f.shape
    fnew = np.zeros_like(f)
    for i in prange(nx):
        v_old = v + E[i] * dt_half
        fnew[i, :] = interp_v_line(f[i, :], v_old, vmax, dv)
    return fnew

@njit(parallel=True)
def advect_x_numba(f, x, v, Lx, dx, dt_full):
    nx, nv = f.shape
    x_old = np.empty((nx, nv))
    for j in prange(nv):
        for i in range(nx):
            x_old[i, j] = x[i] - v[j] * dt_full
    return interp_x_periodic(f, x_old, Lx, dx)

# ----------------------------
# Bucle temporal (Strang splitting)
# ----------------------------
times = [0.0, 15.0, 20.0, 30.0]   # tiempos de la Figura 4.2
snapshots = {0.0: f.copy()}

t = 0.0
while t < tmax + 1e-12:
    rho = rho_from_f(f)
    E = solve_poisson(rho)

    # medio paso en v
    f = advect_v_numba(f, E, v, vmax, dv, 0.5 * dt)

    # paso completo en x
    f = advect_x_numba(f, x, v, Lx, dx, dt)

    # otro medio paso en v
    rho = rho_from_f(f)
    E = solve_poisson(rho)
    f = advect_v_numba(f, E, v, vmax, dv, 0.5 * dt)

    t += dt

    for T in times[1:]:
        if abs(t - T) < 0.5 * dt and T not in snapshots:
            snapshots[T] = f.copy()

# ----------------------------
# Gráficas f(x,v,t) como en Figura 4.2
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
times_plot = [0.0, 15.0, 20.0, 30.0]

for ax, T in zip(axes.flatten(), times_plot):
    snap = snapshots[T]
    im = ax.imshow(
        snap.T,
        extent=[0, Lx, -vmax, vmax],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    ax.set_title(f"t = {T}")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
