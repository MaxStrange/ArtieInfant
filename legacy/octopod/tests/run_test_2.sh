# This test should be run last
TOPIC1='input_topic'
TOPIC2='output_topic'
KAFKA_SERVER='10.0.0.11:9092'
HDFS_HOSTS='10.0.0.178:50070'
python test2.py --hdfshosts=$HDFS_HOSTS --hdfsuname='max' --hdfstmpdir='octotest' --producer_topics $TOPIC1 --producer_configs bootstrap-server=$KAFKA_SERVER
