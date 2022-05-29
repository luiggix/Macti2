#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:26:55 2018

@author: luiggi
"""
#-----------------------------------------------------------
# Ruta biblioteca macti
#
import os, sys
sys.path.insert(0, os.path.abspath('../../'))
#print(sys.path)
#-----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import macti_lib.visual
from colorama import Fore

def f(A,b,x,c):
    xT = x.reshape(2,1)
    return 0.5 * np.dot(x, np.dot(A, xT)) - np.dot(b, xT) + c

def plotEingen(ax, sol, eve, xg, yg, z):
    xorig = [sol[0], sol[0]]
    yorig = [sol[1], sol[1]]
    u = [eve[0][0], eve[0][1]]
    v = [eve[1][0], eve[1][1]]
    ax.quiver(xorig, yorig, u, v,color='royalblue',scale=10, label='Eigenvectores', zorder=5)
        
def plotLinsys(sistema, zoom=False):
    if sistema == 'Positivo Definido':
        x = np.linspace(-10,10,20)
        A = np.array([[3,2],[2,6]])
        b = np.array([2,-8])
    elif sistema == 'Positivo Indefinido':
        x = np.linspace(-10000,15000,20)
        A = np.array([[3,2],[6,3.999]])
        b = np.array([2,-8])
    elif sistema == 'Indefinido':
        if zoom:
            x = np.linspace(-10000,10000,30)
        else:
            x = np.linspace(0,1500,30)
        A = np.array([[-.1, 1],[-.9, 3]])
        b = np.array([200,60])
    # elif sistema == 'Indefinido':
    #     x = np.linspace(-10000,10000,50)
    #     A = np.array([[-.1, 1],[-.9, 3]])
    #     b = np.array([200,60])
    
    # Fórmulas de cada línea
    l1 = -A[0,0] * x / A[0,1] + b[0]/A[0,1]
    l2 = -A[1,0] * x / A[1,1] + b[1]/A[1,1]

    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax1.plot(x, l1, lw = 5.0, label = 'Linea 1')
    ax1.plot(x, l2, lw = 3.0, label = 'Linea 2')

    try:
        sol = np.linalg.solve(A,b)
    # Punto de cruce de las líneas rectas
        if sistema != 'Positivo Indefinido':
            ax1.scatter(sol[0], sol[1], fc = 'C3', ec ='k', s = 100, 
                        alpha=0.85, zorder=5, label='Solución')
            ax1.set_title('Cruce de las rectas: ({:5.4}, {:5.4})'.format(sol[0], sol[1]))
            ax1.vlines(sol[0], 0, sol[1], ls='--', lw=1.0, color='gray')
            ax1.hlines(sol[1], 0, sol[0], ls='--', lw=1.0, color='gray')
    except np.linalg.LinAlgError as info:
        print(Fore.RESET + 80*'-')
        print(Fore.RED + 'ERROR: \n {}'.format(info))
        print(Fore.RESET + 80*'-')  
#    else:
#            print(Fore.GREEN + '¡Tu resultado es correcto!')
    
    size_grid = len(x)
    y = np.linspace(min(min(l1), min(l2)), max(max(l1), max(l2)), size_grid)
    xg,yg = np.meshgrid(x,y)
    z = np.zeros((size_grid, size_grid))
    for i in range(size_grid):
        for j in range(size_grid):
            xe = np.array([xg[i,j],yg[i,j]])
            z[i,j] = f(A,b,xe,0)
            
    eva, eve = np.linalg.eig(A)
#        plotEingen(ax1, sol, eve, xg, yg, z)  
    ax1.legend(loc='best', bbox_to_anchor=(0.75, 0.5, 0.5, 0.5))
        
    cont = ax1.contour(xg,yg,z,30,cmap='binary') 
    zsol = f(A,b,sol,0)
    surf = ax2.plot_surface(xg, yg, z, cmap='coolwarm', alpha=0.35, antialiased=False)
    ax2.scatter(sol[0], sol[1], zsol, marker='o', color='k', s=50)
    vzsol = [zsol[0] for i in x]
    ax2.plot(x,l1,vzsol, c='k', lw=3.0)
    ax2.plot(x,l2,vzsol, lw=3.0)
    ax2.view_init(20, -40)
    ax2.set_title('Forma cuadrática $f(x)$')
#    ax2.set_zlabel('$f(x)$')


if __name__ == '__main__':
    
    plotLinsys('Positivo Indefinido')
    plt.show()
    
    plotLinsys('Positivo Definido')
    plt.show()

    plotLinsys('Indefinido')
    plt.show()

    plotLinsys('Indefinido', True)
    plt.show()
