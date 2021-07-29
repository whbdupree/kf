from setuptools import setup, find_packages

INSTALL_REQUIREMENTS = ['numpy',
                        'jaxlib',
                        'jax',
                        'matplotlib',
                        ]
setup(name='kf',
      version = 'dev1',
      install_requires = INSTALL_REQUIREMENTS,
      packages=['kf']
      )
