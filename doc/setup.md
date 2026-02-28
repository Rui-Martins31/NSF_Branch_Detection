# Setup

## First steps

Create a virtual environment where you will be installing all the dependencies:
```
python3 -m venv .venv
```

Once the virtual environment is created, you have to source it:
```
source .venv/bin/activate
```

Every time you want to run this script, the virtual environment must be sourced.

## Dependencies

To install all the necessary dependencies, run the following command:
```
pip install -r requirements.txt
```

You only need to run this command once.

## Checkpoints

Run this command in the terminal to download the model's checkpoints.
```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

You only need to run this command once.

## Run

You can start running the script by using the command:
```
python3 main.py
```