# Stork-A

:warning: **Important: This repository is for manuscript review purposes only. It has the exact same source code as the official repository under [github.com/eipm/stork-a](https://github.com/eipm/stork-a) and will become publicly available after manuscript review.**

[![Actions Status](https://github.com/eipm/stork-a/workflows/Docker/badge.svg)](https://github.com/eipm/stork-a/actions) [![Github](https://img.shields.io/badge/github-1.0.0-green?style=flat&logo=github)](https://github.com/eipm/stork-a) [![EIPM Docker Hub](https://img.shields.io/badge/EIPM%20docker%20hub-1.0.0-blue?style=flat&logo=docker)](https://hub.docker.com/repository/docker/eipm/stork-a) [![GitHub Container Registry](https://img.shields.io/badge/GitHub%20Container%20Registry-1.0.0-blue?style=flat&logo=docker)](https://github.com/orgs/eipm/packages/container/package/stork-a) [![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Set up local environment and install dependencies

Create a local environment
`python -m venv src/env`

Activate the virtual environment
(Required every time you want to access the virtual environment)
`source src/env/bin/activate`

Install requirements from requirements.txt
`pip install -r requirements.txt`

## Execute a model as script

You can use the below to run a model. Feel free to edit this file, as this is used only for testing purposes.
`python src/run_as_script.py`

## For executing the API

Visual studio code is already set up to run using the debugger. This is using as default `"USERS_DICT": "{'user1': 'stork'}"`.

To run individually, you can first set the `USERS_DICT` and just run `python src/main.py`
