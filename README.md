# Machine Learning

Below, I present the instructions for the necessary steps to be taken before the classes.


## Hardware requirements

Virtual Machine with Ubuntu 20.04, **RAM** minimum of 5918MB, preferably 8192MB, storage 30GB for convenience 40GB.


## Source code:
```shell 
cd 
git clone https://github.com/int8/ml7-kozm
cd ml7-kozm
```



## Installing venv
It will help us manage Python versions and environments.

First, let's install the packages needed to install the correct versions of Python in the Ubuntu system.
```bash
sudo apt-get update
sudo apt-get install curl python3-venv 
```

This will take a moment

## Creating a Python environment


Let's verify if the version of Python we are using is higher than 3.7.

```shell 
python3 --version 
```


If so, we can proceed to create a virtual environment.

```shell 
python3 -m venv mlcourse 
source mlcourse/bin/activate
```


After activating the environment, your prompt will inform you about the virtual environment.

Now we install the required packages.
```shell
pip install -r requirements.txt 
```


## Starting (and resuming work)
Whenever resuming work (restarting the computer and the virtual machine), please go to:
```bash
cd $HOME/ml7-kozm
```
activate the environment named *mlcourse*:
```bash
source mlcourse/bin/activate 
```
Then, call the **Jupyter Notebook** tool
```bash
jupyter notebook
```