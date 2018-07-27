from setuptools import setup

__version__ = ['0.0']

setup(name='wildfire',
      version='.'.join(__version__),
      description='Various Implementations of neural models defined in papers',
      url='http://gitlab.com/theSage21/wildfire',
      author='Arjoonn Sharma',
      author_email='arjoonn.94@gmail.com',
      packages=['wildfire'],
      install_requires=['pytorch'],
      keywords=['wildfire', 'pytorch', 'paper implementation'],
      zip_safe=False)
