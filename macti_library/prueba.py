from macti import evaluacion
import numpy as np
x = np.linspace(0,1500,10)
PA = 0.10 * x + 200
PB = 0.35 * x + 20

d = evaluacion.Evalua('../recursos/datos/SistemasLineales/')
d.verifica(PA, 1)
d.verifica(PB, 2)
                            
