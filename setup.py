import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="manix",
    version="0.0.1",
    author="Hooman Karamnejad",
    packages=["manix"],
    description="Using Machine Learning to Detect Phase Transition on Ising Model",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/hoomania/phase_transition_machine_learning.git",
    license='MIT',
    python_requires='>=3.8',
    install_requires=[]
)
