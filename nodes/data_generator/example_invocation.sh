SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python $SCRIPT_DIR/data_generator.py ~/Dropbox/thesis/data/2018/ --producer_topics audiosegments --producer_configs bootstrap-server=10.0.0.11:9092 acks='all' compression-type='lz4' retries=5 buffer-memory=268435456 max-request-size=268435456 retry-backoff-ms=2000 max-in-flight-requests-per-connection=1 --loglevel DEBUG --slice_length 5 --remove_silence
