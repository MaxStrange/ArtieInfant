"""
Test for myargparse library.
"""
import myargparse

if __name__ == "__main__":
    consumer_topics, prod_topics, consumer_configs, prod_configs = myargparse.parse_args()
    print("Consumer topics:", consumer_topics)
    print("Producer topics:", prod_topics)
    print("Consumer configs:", consumer_configs)
    print("Producer configs:", prod_configs)

