from distutils.core import setup, Extension
import sys, os

version = file('VERSION.txt').read().strip()


setup(name='codemaker',
      version=version,
      description="Embedding and sparse coding of dense and/or high dimensional data",
      long_description=file('README.rst').read(),
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords=('machine-learning artificial-intelligence scientific numerical'
                'artificial-neural-networks'),
      author='Olivier Grisel',
      author_email='olivier.grisel@ensta.org',
      url='http://github.com/ogrisel/codemaker',
      license='MIT',
      package_dir={'': 'src'},
      packages=['codemaker'])
