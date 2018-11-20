#!/bin/bash
set -e
if [ -d ./env3 ]; then
  echo 'Environment found, starting program.'
else
  echo 'Environment not found, initialising.'
  python3 -m venv env3/
fi
source env3/bin/activate
echo "Installing missing python packages."
pip install -r ./exe/requirements.txt
python3 source/solution.py "$@" -i examples