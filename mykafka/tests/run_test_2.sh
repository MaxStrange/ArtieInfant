# This test should be run last
TOPIC1='input_topic'
TOPIC2='output_topic'
KAFKA_SERVER='10.0.0.11:9092'
python3 test2.py --producer_topics $TOPIC1 --producer_configs bootstrap-server=$KAFKA_SERVER
