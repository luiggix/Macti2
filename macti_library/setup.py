from setuptools import find_packages, setup
setup(
    name='macti',
    packages=find_packages(include=['macti', 'macti.SistemasLineales', 'macti.MetodoEuler']),
    version='0.1.0',
    description='Macti 2.0',
    author='Luis M. de la Cruz',
    license='CC',
)
