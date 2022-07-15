# Herramientas para colorear texto y comparación de los resultados
from colorama import Fore
from nose.tools import assert_equal
import numpy as np

class Evalua():
    def __init__(self, path):
        self.path = path
        
    def verifica(self, x, i):
        """
        Permite comparar el contenido de x con el de y. Si se encuentra una diferencia entonces emite una alerta.
        """
        y = np.load(self.path + 'sol{:02d}.npy'.format(i))

        try:
            assert_equal(list(x.flatten()), list(y.flatten()))
        except AssertionError as info:
            print(Fore.RESET + 80*'-')
            print(Fore.RED + 'Cuidado: Ocurrió un error en tus cálculos: \n {}'.format(info))
            print(Fore.RESET + 80*'-')
        else:
            print(Fore.GREEN + '¡Tu resultado es correcto!')

#----------------------- TEST OF THE MODULE ----------------------------------   
if __name__ == '__main__':
    x = np.linspace(0,1500,10)
    PA = 0.10 * x + 200
    PB = 0.35 * x + 20

    d = Evalua('../datos/SistemasLineales/')     
    d.verifica(PA, 1)
    d.verifica(PB, 2)
