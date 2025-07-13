from setuptools import setup, find_packages
import pkg_resources

with open("requirements.txt", "r") as f:
    requirements = [str(req) for req in pkg_resources.parse_requirements(f)]

setup(name="langchain-app",
      version="0.0.1",
      packages=find_packages(),
      install_requires=requirements)