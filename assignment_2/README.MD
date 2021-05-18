# Assignment 3 - Edge Detection

## Prerequisites
You will need to have Bash and Python 3 installed on your device. This script has been tested with Python 3.8.6 on Linux (Ubuntu flavour).
As this has only been tested on Linux, I would recommend running the script on Worker2 or on Windows Subsystem for Linux (Ubuntu) if you are on Windows as the following instructions are written for Linux.

## Installation
- Clone this repository somewhere on your device
- Open a Bash terminal in `/assignment_2/` of the cloned repository or `cd assignment_2` from the root of the repository
- Run the Bash script to generate your virtual environment, generate required directories, and install dependencies:

```bash
./create_venv_unix.sh
```
- If you have any issues with the last step, make sure that you can execute the bash scripts with the following command and try again:

```bash
chmod +x create_venv_unix.sh
```

**NOTE:** Every command from this point will be executed from inside `/assignment_2/`

## Running the script
- Make sure the newly created virtual environment is activated with:

```bash
source assignment_2_venv/bin/activate
```
---------------------------------------------------------

Your directory should now have at least these files and directories:

```
.
├── assignment_2_venv
├── create_venv_unix.sh
├── data
├── edge_detection.py
├── output
└── README.MD
```

**NOTE:** Going forwards I will assume you have an alias set for running `python` such that you will not have to type `python3` if that is normally required on your system (usually Mac / Linux). If this is not the case, you will have to substitute `python` with `python3` in the following commands.

- `./data/` should contain an image from the Jefferson Memorial
    - This image is already included in this repository. Alternatively the image can be downloaded from [this link](https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG) as such:
        - `wget https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG -O ./data/memorial.jpg` - Download image and save in `./data/`
- Run the script `image_search.py` and supply a target filename as the first argument
- Optionally change the directory containing image files with the flag `-d` or `--data_path`
- E.g.:
```bash
# default data_path
python image_search.py image_0002.jpg

# own data_path
python image_search.py image_0002.jpg -d ./some/path/
```
- Your terminal should show a readout of the script's progression and finally a note about which image in `data_path` that most closely resembles the target image
- `./output/` should now contain a `.csv`-file (named after the target image) with a table of the `CHISQR` histogram distances to every other image in the `data_path` ordered from close to distant
    - This directory should already have an example output file when cloning the directory
