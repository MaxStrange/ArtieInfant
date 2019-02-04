SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python $SCRIPT_DIR/reducer.py --consumer_configs bootstrap-server=$KAFKA_SERVER fetch-max-bytes=268435456 max-partition-fetch-bytes=268435456 retry-backoff-ms=2000 max-in-flight-requests-per-connection=1 --consumer_topics resampled
