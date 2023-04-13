from setuptools import find_packages,setup
from typing import List

E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    Return list of project requirements.
    '''
    #requirements = []
    with open(file_path) as file_obj:
        requirements_file = file_obj.readlines()
        requirements =  requirements_file.split('\n')

        if E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
name = 'CovertypeClassification',
version = '0.0.1',
author = 'Marcin',
author_email = 'mbierbasz@op.pl',
packages = find_packages(),
install_requires =get_requirements('requirements.txt')
)