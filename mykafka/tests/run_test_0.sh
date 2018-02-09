# This test should be run first
TOPIC1='input_topic'
TOPIC2='output_topic'
KAFKA_SERVER='10.0.0.11:9092'
python3 test0.py --consumer_topics $TOPIC1 --consumer_configs bootstrap-server=$KAFKA_SERVER
