# This test should be run second
TOPIC1='input_topic'
TOPIC2='output_topic'
KAFKA_SERVER='10.0.0.11:9092'
python3 test.py --consumer_topics $TOPIC1 --producer_topics $TOPIC2 --consumer_configs bootstrap-server=$KAFKA_SERVER
