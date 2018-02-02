# This test should be run last
TOPIC1='input_topic'
TOPIC2='output_topic'
../kafka_2.12-1.0.0/bin/kafka-console-producer.sh --broker-list localhost:9092 --topic $TOPIC1
