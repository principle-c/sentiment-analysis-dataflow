#[START load_dependencies]
import setuptools



setuptools.setup(
    name='pytorch-sentiment-prediction',
    version='v1',
    requirements_file='requirement.txt',
    packages=setuptools.find_packages(),
)
#[END load_dependencies]