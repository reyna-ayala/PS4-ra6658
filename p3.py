import os
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix, coo_matrix


def lax_wendroff(a,b,dx,dy,dt,N):
    i = np.arange(N)
    j = np.arange(N)
    ii, jj = np.meshgrid(i,j,indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    k = ii*N + jj
    
    # calculate +/- indices from the current i,j
    ip = ((ii+1)%N *N) + jj
    im = ((ii-1)%N *N) + jj
    jp = ii*N + (jj+1)%N
    jm = ii*N + (jj-1)%N
    
    ipjp = ((ii+1)%N *N) + (jj+1)%N
    imjp = ((ii-1)%N *N) + (jj+1)%N
    ipjm = ((ii+1)%N *N) + (jj-1)%N
    imjm = ((ii-1)%N *N) + (jj-1)%N
            
    rows = np.concatenate([k]*9)
    cols = np.concatenate([k, ip, im, jp, jm, ipjp, imjp, ipjm, imjm])
    vals = np.concatenate([
        np.full(len(k), -(a**2 * dt**2)/dx**2 - (b**2 * dt**2)/dy**2 + 1),
        np.full(len(k), -b*dt/(2*dx) + (b**2 * dt**2)/(2*dx**2)),
        np.full(len(k), b*dt/(2*dx) + (b**2 * dt**2)/(2*dx**2)),
        np.full(len(k), -a*dt/(2*dy) + (a**2 * dt**2)/(2*dy**2)),
        np.full(len(k), a*dt/(2*dy) + (a**2 * dt**2)/(2*dy**2)),
        np.full(len(k), (a*b*dt**2)/(4*dx*dy)),
        np.full(len(k), -(a*b*dt**2)/(4*dx*dy)),
        np.full(len(k), -(a*b*dt**2)/(4*dx*dy)),
        np.full(len(k), (a*b*dt**2)/(4*dx*dy)),
    ])
    
    vals = vals.astype(np.float32)

    return coo_matrix((vals, (rows,cols)), shape=(N**2,N**2)).tocsr()

def ctu(a,b,dx,dy,dt,N):
    
    i = np.arange(N)
    j = np.arange(N)
    ii, jj = np.meshgrid(i,j,indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    k = ii*N + jj
    
    # calculate +/- indices from the current i,j
    im = ((ii-1)%N *N) + jj
    jm = ii*N + (jj-1)%N
    
    imjm = ((ii-1)%N *N) + (jj-1)%N

    m = a*dt/dx
    v = b*dt/dy
    
    rows = np.concatenate([k]*4)
    cols = np.concatenate([k, im, jm, imjm])
    vals = np.concatenate([
        np.full(len(k), (1-m)*(1-v)),
        np.full(len(k), (1-v)*m),
        np.full(len(k), v*(1-m)),
        np.full(len(k), v*m),
    ])

    vals = vals.astype(np.float32)
    
    return coo_matrix((vals, (rows,cols)), shape=(N**2,N**2)).tocsr()

'''
@njit
def flatten_array_2D(m):
    sz = m.size
    shp = m.shape
    
    m_flat = np.zeros((sz))
    
    for i in range(shp[0]):
        for j in range(shp[1]):
            k = i*N + j
            m_flat[k] = m[i,j]
            
    return m_flat

@njit
def flatten_array_3D(m,Nt):
    sz = int(m.size / Nt)
    shp = m.shape
    
    m_2D = np.zeros((Nt, sz))
    
    for i in range(shp[0]):
        for j in range(shp[1]):
            k = i*N + j
            m_2D[:,k] = m[i,j,:]
            
    return m_2D
'''

def unflatten(m_flat):
    sz = m_flat.size
    N = int(np.sqrt(sz))
    m = np.reshape(m_flat, (N,N), order='C')
    return m

'''
def exact_sol(x,y,t,a,b):
    x_tile, y_tile, t_tile = np.meshgrid(x,y,t)
    
    xp_tile = (x_tile - a*t_tile) % 1
    yp_tile = (y_tile - b*t_tile) % 1
    
    return np.exp( -( (xp_tile-0.5)**2 + (yp_tile-0.5)**2 ) / (3/20)**2 )
'''

def L2norm(u_approx, u_exact):
    return np.sum( np.sqrt((u_exact - u_approx)**2) )


N = int(input('N = '))

a = 1
b = 2

x = np.linspace(0,1,N); dx = x[1] - x[0]
y = np.linspace(0,1,N); dy = y[1] - y[0]

dt = dx/(a+b) * 0.25
t = np.arange(0,10+dt, dt)

x_tile, y_tile = np.meshgrid(x,y)

#u_exact_array = exact_sol(x,y,t,a,b)
#u_exact = u_exact_array.ravel()

idx_t1 = [min(range(len(t)), key=lambda i: abs(t[i]-1))][0]
idx_t10 = [min(range(len(t)), key=lambda i: abs(t[i]-10))][0]


# LAX WENDROFF ---------------------------------------------

LW = lax_wendroff(a,b,dx,dy,dt,N)
L2_LW = np.zeros(t.shape)

u_init = np.exp( -( (x_tile-0.5)**2 + (y_tile-0.5)**2 ) / (3/20)**2 )
u = u_init.ravel().astype(np.float32)

for idx in range(len(t)):

    xp = (x_tile - a*t[idx]) % 1
    yp = (y_tile - b*t[idx]) % 1
    
    u_exact_idx =  np.exp( -( (xp-0.5)**2 + (yp-0.5)**2 ) / (3/20)**2 ).ravel()
    L2_LW[idx] = L2norm(u,u_exact_idx)

    if idx == idx_t1:
        u_t1 = unflatten(u)
    elif idx == idx_t10:
        u_t10 = unflatten(u)

    u = LW @ u

print('LW done.')

ut1_LW = pd.DataFrame(u_t1)
ut10_LW = pd.DataFrame(u_t10)
err_LW = pd.DataFrame(L2_LW)

ut1_LW.to_csv(f'ut1_LW_N{N}.csv')
ut10_LW.to_csv(f'ut10_LW_N{N}.csv')
err_LW.to_csv(f'err_LW_N{N}.csv')

print('LW CSVs written.')


# CTU  ---------------------------------------------

CTU = ctu(a,b,dx,dy,dt,N)
L2_CTU = np.zeros(t.shape)

u = u_init.ravel().astype(np.float32)

for idx in range(len(t)):

    xp = (x_tile - a*t[idx]) % 1
    yp = (y_tile - b*t[idx]) % 1
    
    u_exact_idx =  np.exp( -( (xp-0.5)**2 + (yp-0.5)**2 ) / (3/20)**2 ).ravel()
    L2_CTU[idx] = L2norm(u,u_exact_idx)

    if idx == idx_t1:
        u_t1 = unflatten(u)
    elif idx == idx_t10:
        u_t10 = unflatten(u)

    u = CTU @ u

print('CTU done.')

ut1_CTU = pd.DataFrame(u_t1)
ut10_CTU = pd.DataFrame(u_t10)
err_CTU = pd.DataFrame(L2_CTU)

ut1_CTU.to_csv(f'ut1_CTU_N{N}.csv')
ut10_CTU.to_csv(f'ut10_CTU_N{N}.csv')
err_CTU.to_csv(f'err_CTU_N{N}.csv')

print('CTU CSVs written.')
