"""
This module is a test file for saving a file.
"""
import os

def main(fcontents):
    testpath = 'blahpath_definitely_a_test_just_a_test.wav'
    with open(testpath, 'wb') as f:
        f.write(fcontents)
    os.remove(testpath)
    return "saved!"
