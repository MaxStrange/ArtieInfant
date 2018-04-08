"""
This test script runs a while loop forever until it is killed.
"""
import time

def main():
    sum = 0
    while True:
        sum += 1
        time.sleep(1)
    return sum
