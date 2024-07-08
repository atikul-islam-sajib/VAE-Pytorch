from setuptools import setup, find_packages


def requirements():
    with open("./requirements.txt", "r") as file:
        return file.read().splitlines()


setup(
    name="VAE-Pytorch",
    version="0.1.0",
    description="In machine learning, a variational autoencoder is an artificial neural network architecture introduced by Diederik P. Kingma and Max Welling. It belongs to the family of probabilistic graphical models and variational Bayesian methods",
    author="Atikul Islam Sajib",
    author_email="atikul.sajib@ptb.de",
    url="https://github.com/atikul-islam-sajib/VAE-Pytorch.git",  # Update with your project's GitHub repository URL
    packages=find_packages(),
    install_requires=requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="VAE-Pytorch machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/VAE-Pytorch.git/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/VAE-Pytorch.git",
        "Source Code": "https://github.com/atikul-islam-sajib/VAE-Pytorch.git",
    },
)