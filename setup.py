from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

# file from k-gnn to build package graph_cpu for baseline

__version__ = '0.0.0'
url = 'https://github.com/vthost/ogb-revisited'

install_requires = []
setup_requires = []  #'pytest-runner']
tests_require = []  #'pytest', 'pytest-cov']

ext_modules = [CppExtension('graph_cpu', ['cpu/graph.cpp'])]
cmdclass = {'build_ext': BuildExtension}

setup(
    name='k_gnn',
    version=__version__,
    description='',
    author='XXX',
    author_email='XXX',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
