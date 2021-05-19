# Assignment 4 - Classification Benchmarks

## Prerequisites
You will need to have Bash and Python 3 installed on your device. This script has been tested with Python 3.8.6 on Linux (Ubuntu flavour).
As this has only been tested on Linux, I would recommend running the script on Worker2 or on Windows Subsystem for Linux (Ubuntu) if you are on Windows as the following instructions are written for Linux.

## Installation
- Clone this repository somewhere on your device
- Open a Bash terminal in `/assignment_4/` of the cloned repository or `cd assignment_4` from the root of the repository
- Run the Bash script to generate your virtual environment, generate required directories, and install dependencies:

```bash
./create_venv_unix.sh
```
- If you have any issues with the last step, make sure that you can execute the bash scripts with the following command and try again:

```bash
chmod +x create_venv_unix.sh
```

**NOTE:** Every command from this point will be executed from inside `/assignment_4/`

## Running the scripts
- Make sure the newly created virtual environment is activated with:

```bash
source assignment_4_venv/bin/activate
```

Your directory should now have at least these files and directories:

```
.
├── assignment_4_venv/
├── create_venv_unix.sh
├── data/
├── lr-mnist.py
├── nn-mnist.py
├── output/
├── README.MD
├── requirements.txt
└── utils/
```

**NOTE:** Going forwards I will assume you have an alias set for running `python` such that you will not have to type `python3` if that is normally required on your system (usually Mac / Linux). If this is not the case, you will have to substitute `python` with `python3` in the following commands.

- Run the script `lr-mnist.py` - this can take a while
    - You can optionally supply the flag `-d` or `--data_path` to set the directory to save / load the MNIST dataset (default is `./data/`)
    - E.g.:

    ```bash
    # default data_path
    python lr-mnist.py

    # custom data_path
    python lr-mnist.py -d ./some/path
    ```

    - If the MNIST dataset is not present in `data_path`, it will be downloaded, otherwise it will use the existing files
- Your terminal should output performance metrics of the logistic regression classifier as well as save them to `./output/lr_classifier_metrics.txt`
    - An example file should already be present when cloning this repository
- Run the script `nn-mnist.py` - this can take a while
    - You can optionally supply the flag `-d` or `--data_path` to set the directory to save / load the MNIST dataset (default is `./data/`)
    - You can specify the number of epochs to train with `-e` or `--epochs` (default is 5)
    - E.g.:

    ```bash
    # default data_path
    python nn-mnist.py

    # custom data_path
    python n-mnist.py -d ./some/path -e 100
    ```
    - If the MNIST dataset is not present in `data_path`, it will be downloaded, otherwise it will use the existing files
- Your terminal should output performance metrics of the neural network classifier as well as save them to `./output/nn_classifier_metrics.txt`
    - An example file (run with 5 epochs) should already be present when cloning this repository