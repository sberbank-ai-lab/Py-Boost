#!/bin/bash

python -m venv py_boost_venv
source ./py_boost_venv/bin/activate

pip install -U pip
pip install -U poetry

poetry lock
poetry install
poetry build