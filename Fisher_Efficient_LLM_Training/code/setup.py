from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="Subspace Projected Gradient Descent",
    version="1.0",
    description="Memory-Efficient LLM Training by Random Subspace Projection and Natural Gradients",
    url="https://github.com/das-projects/iclr25.git",
    author="Arijit Das",
    author_email="arijit.das@selfsupervised.de",
    license="Apache 2.0",
    packages=["galore_torch"],
    install_requires=required,
)