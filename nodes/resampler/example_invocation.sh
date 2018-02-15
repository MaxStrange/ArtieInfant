TOPIC1='audiosegments'
TOPIC2='resampled'
KAFKA_SERVER='10.0.0.11:9092'
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python $SCRIPT_DIR/resampler.py --consumer_topics $TOPIC1 --producer_topics $TOPIC2 --consumer_configs bootstrap-server=$KAFKA_SERVER fetch-max-bytes=268435456 max-partition-fetch-bytes=268435456 retry-backoff-ms=2000 max-in-flight-requests-per-connection=1

--producer_configs bootstrap-server=$KAFKA_SERVER acks='all' compression-type='lz4' retries=5 buffer-memory=268435456 max-request-size=268435456 retry-backoff-ms=2000 max-in-flight-requests-per-connection=1
