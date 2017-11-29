#!/bin/bash

virtualenv env_ner
source env_ner/bin/activate
pip install -r requirements.txt
mkdir logs
mkdir logs/input
mkdir logs/output