if [ -d ./env3 ]; then
  echo 'Environment found, starting program.'
  source env3/bin/activate &&
  python3 src/solution.py "$@"
else
  echo 'Environment not found, initialising.'
  mkdir env3 &&
  python3 -m venv env3/ &&
  source env3/bin/activate &&
  pip install -r requirements.txt
  echo 'TO START: ./run.sh --help'
fi
