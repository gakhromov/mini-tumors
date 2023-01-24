# Instructions for the local setup
[<< go back to the main README](../README.md)
1. Install Python 3.10.4 via [pyenv](https://github.com/pyenv/pyenv) or the [offical website](https://www.python.org/downloads/). It is recommended to download the python version using `pyenv`, as there could be multiple versions of python already installed to your computer.
2. Install [pipenv](https://github.com/pypa/pipenv) using `pip install pipenv`. This is a python package manager that will setup the code environment and download the necessary packages for the project.
3. Install necessary packages by typing `pipenv install`. This will automatically install all the necessary versions for all packages required to run the project.
4. The environment should be ready! You can use the environment in two ways:
    - You can run one command in the environment, leaving your terminal outside of the environment. To do so, type `pipenv run <your command>` (your command should be without angular brackets. For example: `pipenv run python main.py`)
    - You can also convert your terminal to use the environment. In this case, all subsequent commands will see the necessary python packages. You can do that by typing `pipenv shell`. You can then execute your commands normally. To exit the environment, type `exit`.

# Instructions for the Euler setup
[<< go back to the main README](../README.md)
1. Access the Euler cluster via SSH connection. See [this tutorial](https://scicomp.ethz.ch/wiki/Accessing_the_clusters). I recommend using a program that provides some kind of GUI so that you can easily navigate the cluster filesystem (I personally use MobaXterm). You need to be on the ETH network (directly or via VPN) to access Euler.
2. Type the command `module load gcc/8.2.0 python_gpu/3.10.4`. This will load the necesary python version to your current session. **NB**: you can avoid having to enter this command everytime you access the cluster by appending those lines (`module load gcc/8.2.0 python_gpu/3.10.4`) to the `.bashrc` file in your home directory on the cluster. 
3. Clone this github repo into your Euler home directory using classic Git commands such as `git clone`.
4. In the cluster, navigate to the repo and type the following commands:
    - Type `pip install pipenv` to install [pipenv](https://github.com/pypa/pipenv). This is a python package manager that will setup the code environment and download the necessary packages for the project.
    - Type `pipenv install`. This will automatically install all the necessary versions for all packages required to run the project.
5. The environment should be ready! You can use the environment in two ways:
    - You can run one command in the environment, leaving your terminal outside of the environment. To do so, type `pipenv run <your command>` (your command should be without angular brackets. For example: `pipenv run python main.py`)
    - You can also convert your terminal to use the environment. In this case, all subsequent commands will see the necessary python packages. You can do that by typing `pipenv shell`. You can then execute your commands normally. To exit the environment, type `exit`.
