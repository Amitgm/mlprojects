from setuptools import find_packages,setup

from typing import List


HYPHEN_E_DOT ="-e ."

def get_requirements(file_path:str)->List[str]:

    """
    this function returns a list of requirements

    """

    requirements = []

    with open(file_path) as file_obj:

        # \n gets recorded over here

        requirements = file_obj.readlines()

        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:

            requirements.remove(HYPHEN_E_DOT)


        return requirements 








setup(

name="mlprojects",
version="0.0.1",
author = "Amit George",
author_email="amittimer@gmail.com",
# find pacakges will see in how many folders has __init__.py, it consider those folders as a package itself and then try to build this
# once built, it can be imported wherever we want
packages=find_packages(),
# when installing packages in form of list -e will also come 
install_requires= get_requirements("requirements.txt")

)