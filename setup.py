from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="src",
    packages=find_packages(include=["src"]),
    version="1.0.0",
    description="PCA Colour Augmentation",
    author="Amrit Purshotam",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "bandit==1.7.0",
            "black==20.8b1",
            "flake8==3.9.2",
            "flake8-bandit==2.1.2",
            "flake8-bugbear==21.4.3",
            "flake8-builtins==1.5.3",
            "flake8-comprehensions==3.5.0",
            "isort==5.9.3",
            "mypy==0.910",
            "rope==0.19.0",
        ]
    },
)
