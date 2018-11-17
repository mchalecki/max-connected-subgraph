#!/bin/bash
set -e
if [ -d ./env3 ]; then
  echo 'Environment found, starting program.'
else
  echo 'Environment not found, initialising.'
  for package in python3-venv python3-tk; do
      if ! dpkg-query -W -f='${Status}' $package | grep "ok installed" &>/dev/null; then
      echo "No $package, installing.
Running command 'sudo apt install $package', require sudo.";
      sudo apt install $package;
      else
        echo "$package detected"
      fi
  done
  python3 -m venv env3/
fi
source env3/bin/activate
echo "Installing missing python packages."
pip install -r requirements.txt
python3 source/solution.py "$@"