# This test should be run last
TOPIC1='input_topic'
TOPIC2='output_topic'
KAFKA_SERVER='tegra:9092'
#../../kafka_2.12-1.0.0/bin/kafka-console-producer.sh --broker-list 10.0.0.11:9092 --topic $TOPIC1
python3 test2.py --producer_topics $TOPIC1 --producer_configs bootstrap-server=10.0.0.11:9092