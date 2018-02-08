# This test should be run first
TOPIC1='input_topic'
TOPIC2='output_topic'
KAFKA_SERVER='localhost:9092'
#../kafka_2.12-1.0.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic $TOPIC1
#../kafka_2.12-1.0.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic $TOPIC2
#../kafka_2.12-1.0.0/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic $TOPIC2 --from-beginning
python3 test0.py --consumer_topics $TOPIC1 --consumer_configs bootstrap-server=$KAFKA_SERVER
