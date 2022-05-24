#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:58:12 2018

@author: luiggi
"""

import numpy as np
import StationarySolvers as sol
import time
from colorama import Fore
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 14

def buildDomain(filename):
    img=mpimg.imread(filename)[:,:,0]
    img_new = np.zeros(img.shape) 
    img_new[img > 0] = 1.0
    img_new[img == 0] = 1.0   
    sx = img_new.shape[0]
    sy = img_new.shape[1]
    forma = (sx * 3, sy * 3)
    domain = np.ones(forma)
    domain[sx:sx*2, sy:sy*2] = img_new[:]
    return np.flip(img_new.T, 1)

def plot_mesh(xm, ym):
    for y in ym:
        plt.plot([xm[0],xm[-1]], [y,y], color = 'gray', ls = '-', lw = 0.5)
    for x in xm:
        plt.plot([x,x], [ym[0],ym[-1]], color = 'gray', ls = '-', lw = 0.5)
        
def plotMalla(xg, yg, xn,yn, Lx, Ly):
    plt.scatter(xg, yg, marker='.')
    plt.xticks(xn, labels=[])
    plt.yticks(yn, labels=[])
    plt.xlabel('Malla')
#    plt.ylabel('$y$')
    lmax = max(Lx,Ly)
    offx = lmax * 0.1
    offy = lmax * 0.1
    plt.xlim(-offx, Lx+offx)
    plt.ylim(-offy, Ly+offy)
    plot_mesh(xn, yn)
    ax = plt.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cax.set_xticks([])
    cax.set_yticks([])
    
def plotSolution(u, Lx, Ly, Nx,Ny,title):
    x = np.linspace(0,Lx,Nx+2)
    y = np.linspace(0,Ly,Ny+2)
#    xg, yg = np.meshgrid(x,y)
    xg, yg = np.meshgrid(x, y, indexing='ij', sparse=False)

    fig = plt.figure(figsize=(10,4))
        
    cf = plt.contourf(xg, yg, u, levels = 50, alpha=.75, cmap="YlOrRd")
#    cl = plt.contour(xg, yg, u, levels = 10, colors='k', linewidths=0.5)
#    plt.clabel(cl, inline=True, fontsize=10.0)
  
#    plt.xticks(x, labels=[])
#    plt.yticks(y, labels=[])
    plt.xlabel('Solución')
    #plt.ylabel('$y$')
    lmax = max(Lx,Ly)
    offx = lmax * 0.1
    offy = lmax * 0.1
    plt.xlim(-offx, Lx+offx)
    plt.ylim(-offy, Ly+offy)
#    plot_mesh(x, y)
    ax = plt.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cax.set_xticks([])
    cax.set_yticks([])
    fig.colorbar(cf, cax=cax, orientation='vertical')
    
#    plt.suptitle(title, color='blue')

def plotFlujo(v, Lx, Ly, Nx,Ny):
    x = np.linspace(0,Lx,Nx+2)
    y = np.linspace(0,Ly,Ny+2)
    xg, yg = np.meshgrid(x, y, indexing='ij', sparse=False)

    fig = plt.figure(figsize=(10,4))
    plt.quiver(xg, yg, v[0], v[1])
#    plt.streamplot(xg, yg, v[0], v[1])

#    plt.xticks(x, labels=[])
#    plt.yticks(yn, labels=[])
    plt.xlabel('$x$')
#plt.ylabel('$y$')
    lmax = max(Lx,Ly)
    offx = lmax * 0.1
    offy = lmax * 0.1
    plt.xlim(-offx, Lx+offx)
    plt.ylim(-offy, Ly+offy)
#    plot_mesh(xn, yn)
    ax = plt.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cax.set_xticks([])
    cax.set_yticks([])

def buildMatrix(Nx, Ny):
    N = Nx * Ny
    A = np.zeros((N,N))

# Primero llena los bloques tridiagonales
    for j in range(0,Ny):
        ofs = Nx * j
        A[ofs, ofs] = -4 
        A[ofs, ofs + 1] = 1 
        for i in range(1,Nx-1):
            A[ofs + i, ofs + i]     = -4 
            A[ofs + i, ofs + i + 1] = 1 
            A[ofs + i, ofs + i - 1] = 1 
        A[ofs + Nx - 1, ofs + Nx - 2] = 1 
        A[ofs + Nx - 1, ofs + Nx - 1] = -4 

# Despues llena las dos diagonales externas
    for k in range(0,N-Nx):
        A[k, Nx + k] = 1 
        A[Nx + k, k] = 1 

    return A

def RHS(Nx, Ny, boundaries):
    f = np.zeros((Ny,Nx)) # RHS
# Aplicacion de las condiciones de frontera Dirichlet
    JJ = int(Nx * 0.25)
    II = int(Ny * 0.25)
    print(Nx, Ny, JJ, II)
    
#    f[Ny-1,:] -= boundaries['TOP'] # Upper wall
    f[0   ,:] -= boundaries['BOT'] # Bottom wall
    f[Ny-1, JJ*2-1:-1   ] -= boundaries['TOP']
#    f[0   ,JJ-1:JJ*3-1] -= boundaries['BOT'] # Bottom wall    

#    f[:,Nx-1] -= boundaries['RIGHT'] # Right wall
#    f[:,0   ] -= boundaries['LEFT'] # Left wall 
    f[II*2-1:II*3-1   ,Nx-1] -= boundaries['RIGHT']   
    f[II*0-0:II*2-1   ,Nx-1] -= boundaries['BOT'] # RIGHT  
    f[II*2-1:II*3-1,0   ] -= boundaries['LEFT'] # Left wall 
    
    f.shape = f.size     # Cambiamos los arreglos a formato unidimensional

    return f

def flujoCalor(u, Nx, Ny):
    qx = np.zeros((Nx+2, Ny+2))
    qy = qx.copy()

    s = 1.0# k / 2*h
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            qx[i,j] = -s * (u[i+1,j] - u[i-1,j])
            qy[i,j] = -s * (u[i,j+1] - u[i,j-1])
            
    return qx, qy

def solucion(Lx, Ly, Nx, Ny, boundaries, metodo):
#    Lx = Lx
#    Ly = Ly
#    Nx = Nx
#    Ny = Ny
    N = Nx * Ny
#    boundaries = boundares

    A = buildMatrix(Nx, Ny) # Matriz del sistema
    b = RHS(Nx, Ny, boundaries)
    
    tol = 1e-6
    max_iter = 200
    w = 1.5

    t1 = time.perf_counter()
    
    if metodo == 'linalg.solve':
        ut = np.linalg.solve(A,b)
        error = 0.0
        it = 1
    elif metodo == 'Jacobi':
        ut,error,it, ea = sol.jacobi(A,b,tol,max_iter)
    elif metodo == 'Gauss-Seidel':
        ut,error,it, ea = sol.gauss_seidel(A,b,tol,max_iter)
    elif metodo == 'SOR':
        ut,error,it, ea = sol.sor(A,b,tol,max_iter,w)

    t2 = time.perf_counter()
    te = t2 - t1
    print(Fore.BLUE + "Método: {}\n".format(metodo) +
          Fore.RESET + " CPU: {:5.4f} [s] ".format(te) +
          Fore.MAGENTA + " Sistema : {}x{} = {}\n".format(N*N, N*N, (N * N)**2) +
          Fore.RED   + " Error : {:5.4e} ".format(error) + 
          Fore.GREEN + " Iter : {} ".format(it))

    u = np.zeros((Ny+2, Nx+2))

    JJ = int(Nx * 0.25)
    II = int(Ny * 0.25)
    print(Nx, Ny, JJ, II)
    
#    u[Ny+1,:   ] = boundaries['TOP']
    u[0   ,:   ] = boundaries['BOT']
    u[Ny+1, JJ*2:-1   ] = boundaries['TOP']
#    u[0   , JJ:JJ*3   ] = boundaries['BOT']

#    u[:   ,Nx+1] = boundaries['RIGHT']
#    u[:   ,0   ] = boundaries['LEFT'] 
    u[II*2:II*3   ,Nx+1] = boundaries['RIGHT']   
    u[II*0:II*2   ,Nx+1] = boundaries['BOT']   
    u[II*2:II*3   ,0   ] = boundaries['LEFT'] 
    
    ut.shape = (Ny, Nx) # Regresamos el arreglo a formato bidimensional
    u[1:Ny+1,1:Nx+1] = ut
    
    titulo = 'Método: {} ][ Error = {:5.4f} | Iter = {} | CPU = {:5.4f} [s] ][ Sistema : {}]'.format(metodo, error, it, te, N * N)
    plotSolution(u.T,Lx, Ly, Nx,Ny,titulo)
    plt.show()

    qx, qy = flujoCalor(u.T, Nx, Ny)
    plotFlujo((qx, qy), Lx, Ly, Nx, Ny)
    plt.show()

    np.save('temperatura.npy', u)
    np.save('flujocalor.npy', (qx, qy))
    
if __name__ == '__main__':
    
    kappa = buildDomain('./Figuras/mactiBN.png')

    print(kappa.shape)
    Lx = 1.0
    Ly = 1.0
    Nx = 81 #kappa.shape[0] - 2
    Ny = 81 #kappa.shape[1] - 2
#    N = Nx * Ny
#    titulo = 'Macti'   
#    plotSolution(kappa, Lx, Ly, Nx, Ny,titulo)
#    plt.show()

    boundaries = {'BOT':40, 'TOP':10, 'LEFT':10, 'RIGHT':30}
    u = solucion(Lx, Ly, Nx, Ny, boundaries, 'linalg.solve')

#    T = 7
#    B = 34
#    L = 100
#    R = 0
#    N = 3 
#    m = 'Jacobi'
#    solucion(B,T,L,R,m,lax, l)
