import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 14

def plotGrid(xm, ym, frame='grid'):
    if frame == 'grid':
        for y in ym:
            plt.plot([xm[0],xm[-1]], [y,y], color = 'gray', ls = '-', lw = 0.5)
        for x in xm:
            plt.plot([x,x], [ym[0],ym[-1]], color = 'gray', ls = '-', lw = 0.5)
    elif frame == 'box':
        plt.plot([xm[0],xm[0]], [ym[0],ym[-1]], color = 'gray', ls = '-', lw = 0.5)
        plt.plot([xm[-1],xm[-1]], [ym[0],ym[-1]], color = 'gray', ls = '-', lw = 0.5)
        plt.plot([xm[0],xm[-1]], [ym[0],ym[0]], color = 'gray', ls = '-', lw = 0.5)
        plt.plot([xm[0],xm[-1]], [ym[-1],ym[-1]], color = 'gray', ls = '-', lw = 0.5)

def calcOffset(xg, yg):
    Lx = xg[-1,0] - xg[0,0]
    Ly = yg[0,-1] - yg[0,0]
    lmin = min(Lx, Ly)
    offx = lmin * 0.1
    offy = lmin * 0.1
    return offx, offy, xg[0,0], xg[-1,0], yg[0,0], yg[0,-1]

def plotMalla(xg, yg, title='', cbar = False, marker='.'):
    plt.scatter(xg, yg, marker=marker)
    x = xg[:,0]
    y = yg[0,:]
    plt.xticks([x[0], x[-1]], labels=[x[0], x[-1]])
    plt.yticks([y[0], y[-1]], labels=[y[0], y[-1]])
    plt.xlabel(title)
    offx, offy, ax, bx, ay, by = calcOffset(xg, yg)
    plt.xlim(ax-offx, bx+offx)
    plt.ylim(ay-offy, by+offy)
    plotGrid(x, y)
    ax = plt.gca()
    ax.set_aspect('equal')
    
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cax.set_xticks([])
        cax.set_yticks([])

def plotContornos(xg, yg, u, title='', frame = 'box', cbar = True):
    cf = plt.contourf(xg, yg, u, levels = 50, alpha=.75, cmap="YlOrRd")
    cl = plt.contour(xg, yg, u, levels = 10, colors='k', linewidths=0.5)
    plt.clabel(cl, inline=True, fontsize=10.0)

    x = xg[:,0]
    y = yg[0,:]
    plt.xticks([x[0], x[-1]], labels=[x[0], x[-1]])
    plt.yticks([y[0], y[-1]], labels=[y[0], y[-1]])
    plt.xlabel(title)
    #plt.ylabel('$y$')
    offx, offy, ax, bx, ay, by = calcOffset(xg, yg)
    plt.xlim(ax-offx, bx+offx)
    plt.ylim(ay-offy, by+offy)
    plotGrid(x, y, frame)
    ax = plt.gca()
    ax.set_aspect('equal')

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cax.set_xticks([])
        cax.set_yticks([])
        fig = plt.gcf()
        fig.colorbar(cf, cax=cax, orientation='vertical')
    
#    plt.suptitle(title, color='blue')

def plotFlujo(xg, yg, u, v, kind='quiver', title='', frame = 'box', cbar=False):
    """
    """

    x = xg[:,0]
    y = yg[0,:]
    plt.xticks([x[0], x[-1]], labels=[x[0], x[-1]])
    plt.yticks([y[0], y[-1]], labels=[y[0], y[-1]])
    plt.xlabel(title)
    #plt.ylabel('$y$')
    offx, offy, ax, bx, ay, by = calcOffset(xg, yg)
    plt.xlim(ax-offx, bx+offx)
    plt.ylim(ay-offy, by+offy)
    
    if kind == 'quiver':
        plt.quiver(xg, yg, u(xg,yg), v(xg,yg), color='gray')
    elif kind == 'stream':
        xg, yg = np.meshgrid(x,y)
        plt.streamplot(x, y, u(xg,yg), v(xg,yg), color='gray', linewidth=0.5)
        
    plotGrid(x, y, frame) 

    ax = plt.gca()
    ax.set_aspect('equal')
    
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cax.set_xticks([])
        cax.set_yticks([])
    
if __name__ == '__main__':

    ax = -3.0
    bx = 3.0
    ay = -3.0
    by = 3.0
    Nx = 21
    Ny = 21
    
    x = np.linspace(ax,bx,Nx+2)
    y = np.linspace(ay,by,Ny+2)
    plotGrid(x,y)
    plt.show()
    
    xg, yg = np.meshgrid(x, y, indexing='ij', sparse=False)
    plotMalla(xg, yg, 'Test ({}x{})'.format(Nx, Ny))
    plt.show()

    plotMalla(xg, yg, 'Test ({}x{})'.format(Nx, Ny), marker='')
    plt.show()
    
    z = (1 - xg/2 + xg**5 + yg**3) * np.exp(-xg**2 - yg**2)
    plotContornos(xg, yg, z, 'Hola', 'grid')
    plt.show()
    
    u = lambda x,y : 1*((-1/2 + 5*x**4) - 2*x*(1-x/2+x**5+y**3)) * np.exp(-x**2-y**2)
    v = lambda x,y : (3*y**2 - 2*x*(1-x/2+x**5+y**3)) * np.exp(-x**2-y**2)
    plotFlujo(xg, yg, u, v, 'quiver', 'flujo', 'box')
    plt.show()

    Nx = 31
    Ny = 11
    x = np.linspace(0.0,3.0,Nx+2)
    y = np.linspace(0.0,1.0,Ny+2)
    xg, yg = np.meshgrid(x, y, indexing='ij', sparse=False)
    A = 1.0
    α = 1.0
    u = lambda x, y: -A * np.cos(α * np.pi * y) * np.sin(α * np.pi * x) 
    v = lambda x, y:  A * np.sin(α * np.pi * y) * np.cos(α * np.pi * x)    
    plotFlujo(xg, yg, u, v, 'quiver', '', 'grid')
    plt.show()