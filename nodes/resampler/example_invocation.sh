TOPIC1='audiosegments'
TOPIC2='resampled'
KAFKA_SERVER='10.0.0.11:9092'
python resampler.py --consumer_topics $TOPIC1 --producer_topics $TOPIC2 --consumer_configs bootstrap-server=$KAFKA_SERVER
