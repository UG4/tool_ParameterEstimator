from setuptools import setup, find_packages

setup(
    name="UGParameterEstimator",
    version="0.1.0",
    url="https://github.com/UG4/ParameterEstimator",
    author="Tim Schön, Moritz Kowalski",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=["numpy", "scipy", "scikit-optimize"],
    extras_require={
        "analysisTool": ["numpy", "scipy", "scikit-optimize", "matplotlib", "PyQt5", "pyqtgraph"],
    },
)
