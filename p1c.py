import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pandas as pd

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve #used for N=512 case

#the root-finding jacobian has boundary conditions of zero

#@njit
def jacobian(u_n, h, N):
    J = lil_matrix((N**2, N**2))
    
    for i in range(N):
        for j in range(N):
            k = i*N + j
            
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                J[k,k] = 1 / h**2
                continue
            
            J[k,k] = (-4 - 4 * h**2 * u_n[k]**3) / h**2
            
            J[k,k-1] = 1 / h**2
            J[k,k+1] = 1 / h**2
            J[k,k-N] = 1 / h**2
            J[k,k+N] = 1 / h**2
    
    return J.tocsr()

@njit
def f(u_n, h, N):
    R = np.zeros((N**2))
    
    for i in range(N):
        for j in range(N):
            k = i*N + j
            
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                R[k] = u_n[k] - 1
                continue
            
            R[k] = (u_n[k-1] + u_n[k+1] + u_n[k-N] + u_n[k+N] - 4*u_n[k]) / h**2 - u_n[k]**4
    
    return R

# PART C: Iterative method

def jacobi(J,f):
    shp = J.shape

    f_new = f.copy()

    maindiag = J.diagonal()

    tol = 1
    counter = 0
    maxiter = 100000

    while tol > 1e-9 and counter < maxiter:

        f_old = f_new.copy()

        f_new = (f - J @ f_old + maindiag * f_old) / maindiag

        tol = np.max( np.abs(f_new - f_old) )
        counter += 1

    Jinv_f = f_new
    #print(counter)
    

    return Jinv_f

N = int(input('N = '))
L = 1
h = L/N

x = np.linspace(0,1,N)
y = np.linspace(0,1,N)

# With Dirichlet, we implement BCs when we set the initial condition
u_old = np.ones((N**2,1)).flatten()

resid = 1
tol = 0.1

while resid > tol:
    fUn = f(u_old, h, N)
    J = jacobian(u_old, h, N)
    Jinv_f = jacobi(J,fUn)
    #Jinv_f = spsolve(J,fUn) #used for N=512 'exact' case
    u_new = u_old - Jinv_f
    resid = np.max(abs(u_new-u_old))
    u_old = u_new
    print(resid)

print('Done.')

u_final = np.reshape(u_new, (N,N))

df = pd.DataFrame(u_final)
df.to_csv(f"ufinal_N{N}.csv")

print('CSV created.')
