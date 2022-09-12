# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

long_description = None
INSTALL_REQUIRES = [
    'cvxpy==1.2.1'
]
EXTRAS_REQUIRE = {
    'local': [
        'matplotlib==3.5',
    ],
}

setup_kwargs = {
    'name': 'sdp-clustering',
    'version': '0.0.1',
    'description': 'Semidefinite programming for clustering',
    'long_description': long_description,
    'license': 'MIT',
    'author': 'Arturo Arranz',
    'url': 'https://github.com/artuntun/SDP-clustering.git',
    'packages': find_packages(),
    'package_data': {},
    'zip_safe': False,
    'install_requires': INSTALL_REQUIRES,
    'extras_require': EXTRAS_REQUIRE,
    'python_requires': '>=3.7',
}


setup(**setup_kwargs)
