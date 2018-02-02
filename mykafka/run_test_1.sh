# This test should be run second
TOPIC1='input_topic'
TOPIC2='output_topic'
python3 test.py --consumer_topics $TOPIC1 --producer_topics $TOPIC2
