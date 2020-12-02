#[START load_dependencies]
from setuptools import setup, find_packages

setup(
    name='pipeline_package',
    author_email='ray.randall@principle-c.com',
    url='na',
    version='0.0.1',
    description='sentiment prediction workflow package, model in pytorch.',
    install_requires=[
        'advertools==0.10.7',
        'google-api-core==1.23.0',
        'google-api-python-client==1.12.8',
        'google-cloud-logging==1.15.1',
        'ipadic==1.0.0',
        'fugashi==1.0.5',
        'mecab-python3==1.0.3',
        'numpy==1.19.2',
        'pandas==1.1.3',
        'pandas-gbq==0.14.1',
        'regex==2020.11.13',
        'torch==1.6.0',
        'transformers==3.5.1',
        'google-cloud-core==1.4.3',
        'google-cloud-storage==1.33.0',
        'google-cloud-bigquery==1.26.1'
    ],
    packages=find_packages(),
    )
#[END load_dependencies]