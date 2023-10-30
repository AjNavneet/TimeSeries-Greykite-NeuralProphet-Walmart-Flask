# Steps_to_install_Neuralprophet_and_Greykite_libraries

Follow these steps to create a new environment and install the necessary libraries:

## Step 1: Create a New Anaconda Environment

```bash
conda create -n myenv python=3.8
```

Replace `myenv` with your preferred environment name and `3.8` with your desired Python version.

## Step 2: Activate the Environment

```bash
conda activate myenv
```

Activate the environment you created in Step 1. You should see your environment name in the command prompt.

## Step 3: Install pip

```bash
conda install pip
```

Make sure you have pip installed within your environment.

## Step 4: Install PyTorch (Version 1.4.0)

```bash
pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp38-cp38-win_amd64.whl
```

Install the specific version of PyTorch you need. Adjust the URL and file name if necessary.

## Step 5: Install NeuralProphet

```bash
pip install neuralprophet
```

Install the NeuralProphet library.

## Step 6: Install Greykite

```bash
conda install -c msys2 libpython m2w64-toolchain
pip install greykite
```

Install the Greykite library by first ensuring you have the required components and then installing Greykite.

That's it! You've now created an environment and installed the libraries you need.
