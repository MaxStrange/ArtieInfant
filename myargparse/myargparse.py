"""
Library that exposes a way to parse what I will need from a user for
every Kafka node.
"""
import argparse

def _parse_dict(d):
    """
    Parses a dict out of `d`, which should be a list of string that contains a list of the form "blah=something".
    """
    def tokenize_and_split(d):
        for token in d:
            if token.strip() != "":
                yield token.split('=')

    return {k: v for (k, v) in tokenize_and_split(d)}

def parse_args():
    """
    Gets all the consumer topic names, producer topic names, and configurations
    for a Kafka node.

    :returns: consumer_names, producer_names, consumer_configs, producer_configs.
              consumer_names and producer_names cannot both be None. If no consumer_configs
              is given, it is returned as an empty dict; same goes for producer_configs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumer_topics", type=str, nargs="*", help="List of names of consumer topics")
    parser.add_argument("--producer_topics", type=str, nargs="*", help="List of names of producer topics")
    parser.add_argument("--consumer_configs", type=str, nargs="*", help="List of property=value strings")
    parser.add_argument("--producer_configs", type=str, nargs="*", help="List of property=value strings")
    args = parser.parse_args()

    if not args.consumer_topics and not args.producer_topics:
        print("consumer_topics and producer_topics cannot both be empty")
        exit(1)
    consumer_configs = _parse_dict(args.consumer_configs) if args.consumer_configs else {}
    producer_configs = _parse_dict(args.producer_configs) if args.producer_configs else {}

    return args.consumer_topics, args.producer_topics, consumer_configs, producer_configs
