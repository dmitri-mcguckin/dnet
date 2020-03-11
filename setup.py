from setuptools import setup, find_packages
from naive_bayes import VERSION

def read(filename):
    with open(filename) as file:
        return file.read()

def get_requirements(filename='requirements.txt'):
    """returns the requirements"""
    requirements = read(filename)
    return list(filter(None, [req.strip() for req in requirements.split() if not req.startswith('#')]))

setup(
    name = 'dnet',
    version = VERSION,
    maintainer = 'dmitri-mcguckin',
    maintainer_email = 'dmitri.mcguckin26@gmail.com',
    description = 'Some neural net stuff.',
    long_description = read('README.md'),
    long_description_content_type = 'text/markdown',
    url = '',
    license = 'MIT',
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'],
    packages = find_packages(),
    scripts=['naive_bayes/src/*', 'neural_net/src/*'],
    install_requires = get_requirements(),
    python_requires = '>=3.6.2'
)