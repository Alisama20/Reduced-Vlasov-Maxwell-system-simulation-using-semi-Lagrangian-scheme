import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm

# ============================================================
# Parámetros
# ============================================================
nx, nvx, nvy = 128, 64, 64   # reducir para pruebas en Windows
Lx = 4*np.pi
pmax_x = 6.0
pmax_y = 6.0

dx = Lx/nx
dpx = 2*pmax_x/nvx
dpy = 2*pmax_y/nvy

dt = 0.02
tmax = 40.0

T = 0.1
alpha = 0.01
k0 = 0.5

x  = np.linspace(0, Lx, nx, endpoint=False)
px = np.linspace(-pmax_x, pmax_x, nvx, endpoint=False)
py = np.linspace(-pmax_y, pmax_y, nvy, endpoint=False)

X  = x[:,None,None]
PX = px[None,:,None]
PY = py[None,None,:]

# ============================================================
# Inicialización
# ============================================================
f0 = (1/(2*np.pi*T)) * np.exp(-(PX**2 + PY**2)/(2*T))
f  = np.tile(f0, (nx,1,1))
f *= (1 + alpha*np.cos(k0*X))

mass0 = np.sum(f)*dx*dpx*dpy

# Potencial vector A_y
A = np.zeros(nx)
A_old = np.zeros(nx)

# ============================================================
# Funciones auxiliares
# ============================================================
def rho_from_f(f):
    return np.sum(f, axis=(1,2)) * dpx*dpy

def J_y_from_f(f):
    gamma = np.sqrt(1 + PX**2 + PY**2)
    J = np.sum(f * (PY/gamma), axis=(1,2)) * dpx*dpy
    return J

def solve_poisson(rho):
    rho -= np.mean(rho)
    k = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    k[0] = 1.0
    rh = np.fft.fft(rho)
    Eh = 1j*rh/k
    Eh[0] = 0.0
    return np.real(np.fft.ifft(Eh))

# ============================================================
# Interpolación cúbica (Catmull-Rom)
# ============================================================
@njit
def cubic_interp_velocity(fline, vnew, vmax, dv):
    nv = fline.shape[0]
    out = np.empty_like(vnew)
    for m in range(vnew.size):
        vv = min(max(vnew[m], -vmax), vmax)
        xi = (vv + vmax)/dv
        i = int(np.floor(xi))
        t = xi - i
        if i < 1: i, t = 1, 0.0
        if i > nv-3: i, t = nv-3, 1.0
        fm1, f0, f1, f2 = fline[i-1], fline[i], fline[i+1], fline[i+2]
        out[m] = 0.5*(2*f0 + (-fm1+f1)*t + (2*fm1-5*f0+4*f1-f2)*t**2 + (-fm1+3*f0-3*f1+f2)*t**3)
    return out

@njit
def cubic_interp_periodic(fline, xnew, L, dx):
    nx = fline.shape[0]
    out = np.empty_like(xnew)
    for m in range(xnew.size):
        xx = xnew[m] % L
        xi = xx/dx
        i = int(np.floor(xi))
        t = xi - i
        im1 = (i-1) % nx
        i0 = i % nx
        i1 = (i+1) % nx
        i2 = (i+2) % nx
        fm1, f0, f1, f2 = fline[im1], fline[i0], fline[i1], fline[i2]
        out[m] = 0.5*(2*f0 + (-fm1+f1)*t + (2*fm1-5*f0+4*f1-f2)*t**2 + (-fm1+3*f0-3*f1+f2)*t**3)
    return out

# ============================================================
# Advección segura Lorentz
# ============================================================
@njit(parallel=True)
def advect_px(f, E_x, B_z, px, py, pmax_x, dpx, dt):
    nx, nvx, nvy = f.shape
    fnew = np.zeros_like(f)
    for i in prange(nx):
        for k in range(nvy):
            px_old = np.empty(nvx)
            for j in range(nvx):
                gamma = np.sqrt(1 + px[j]**2 + py[k]**2)
                Fx = E_x[i] + (py[k]/gamma)*B_z[i]
                px_old[j] = px[j] - Fx*dt
            for j in range(nvx):
                fnew[i,j,k] = cubic_interp_velocity(f[i,:,k], np.array([px_old[j]]), pmax_x, dpx)[0]
    return fnew

@njit(parallel=True)
def advect_py(f, B_z, px, py, pmax_y, dpy, dt):
    nx, nvx, nvy = f.shape
    fnew = np.zeros_like(f)
    for i in prange(nx):
        for j in range(nvx):
            py_old = np.empty(nvy)
            for k in range(nvy):
                gamma = np.sqrt(1 + px[j]**2 + py[k]**2)
                Fy = -(px[j]/gamma)*B_z[i]
                py_old[k] = py[k] - Fy*dt
            for k in range(nvy):
                fnew[i,j,k] = cubic_interp_velocity(f[i,j,:], np.array([py_old[k]]), pmax_y, dpy)[0]
    return fnew

@njit(parallel=True)
def advect_x(f, px, py, x, L, dx, dt):
    nx, nvx, nvy = f.shape
    fnew = np.zeros_like(f)
    for j in prange(nvx):
        for k in range(nvy):
            vx = px[j]/np.sqrt(1 + px[j]**2 + py[k]**2)
            x_old = x - vx*dt
            fnew[:,j,k] = cubic_interp_periodic(f[:,j,k], x_old, L, dx)
    return fnew

# ============================================================
# Bucle temporal
# ============================================================
E_max = []
snap_times = [10,20,60,100,300]
f_maps = {}

nsteps = int(tmax/dt)
t = 0.0

for step in tqdm(range(nsteps)):
    # Campos
    rho = rho_from_f(f)
    E_x = solve_poisson(rho)
    B_z = np.gradient(A, dx)

    # Advección
    f = advect_px(f, E_x, B_z, px, py, pmax_x, dpx, dt)
    f = advect_py(f, B_z, px, py, pmax_y, dpy, dt)
    f = advect_x(f, px, py, x, Lx, dx, dt)

    # Renormalización de masa
    f *= mass0/(np.sum(f)*dx*dpx*dpy)

    # Evolución de A_y (leap-frog)
    J_y = J_y_from_f(f)
    A_new = 2*A - A_old + dt**2 * (np.gradient(np.gradient(A, dx), dx) - J_y)
    A_old, A = A, A_new

    # Guardar max|E_x|
    E_max.append(np.max(np.abs(E_x)))

    # Snapshots
    if any(abs(t-T) < dt/2 for T in snap_times):
        f_maps[t] = np.sum(f, axis=2) * dpy  # f(x, p_x)

    t += dt


# ============================================================
# Gráficas max|E_x|(t)
# ============================================================
plt.figure()
plt.plot(np.arange(len(E_max))*dt, E_max)
plt.xlabel("t")
plt.ylabel("max |E_x|")
plt.title("Campo eléctrico longitudinal |E_x|(t)")
plt.grid()
plt.show()

# ============================================================
# Mapas f(x, p_x) para tiempos seleccionados
# ============================================================
import matplotlib.gridspec as gridspec

# Creamos una figura con tantos subplots como snapshots
n_snap = len(f_maps)
fig = plt.figure(figsize=(15, 3*n_snap))
gs = gridspec.GridSpec(n_snap, 1, hspace=0.4)

for i, T in enumerate(sorted(f_maps.keys())):
    ax = fig.add_subplot(gs[i,0])
    im = ax.imshow(f_maps[T].T, extent=[0,Lx,-pmax_x,pmax_x],
                   origin='lower', aspect='auto', cmap='inferno')
    ax.set_ylabel(r"$p_x$")
    ax.set_title(f"$f(x,p_x)$ at t={T:.1f}")
    if i == n_snap-1:
        ax.set_xlabel("x")
    # Barra de color individual o puedes poner solo una general
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$f(x,p_x)$")

plt.show()

