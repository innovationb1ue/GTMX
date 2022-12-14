from setuptools import setup

setup(
    name='GTMX',
    version='0.0.5',
    description="A Python package for Generative Topographic Mapping (GTM)",
    url='https://github.com/innovationb1ue/GTMX',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'bokeh',
        'pandas'
    ],
    long_description=open('README.rst').read()
)
