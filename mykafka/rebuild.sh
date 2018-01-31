rm -rf build/
rm -rf dist/
rm -rf mykafka.egg-info/
python3 setup.py bdist_wheel
echo "To install, go up one directory and type: 'pip install ./mykafka'"
