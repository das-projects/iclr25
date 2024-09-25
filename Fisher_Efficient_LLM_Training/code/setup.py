from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="subspace_optim",
    version="1.0",
    description="Memory-Efficient LLM Training via Online Subspace Natural Gradient Descent",
    url="https://github.com/das-projects/iclr25.git",
    author="Arijit Das",
    author_email="arijit.das@selfsupervised.de",
    license="Apache 2.0",
    packages=["subspace_optim"],
    install_requires=required,
)
