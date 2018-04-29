"""
This module is the main API for the octopod library.
"""
import contextlib
import logging
import octopod.mykafka as mykafka
import os
import pyhdfs
import tempfile

def init_consumer(*args, **kwargs):
    """
    Must be called before consumer functions can be used.

    Merely a wrapper for calling kafka.KafkaConsumer()
    """
    mykafka.init_consumer(*args, **kwargs)

def init_producer(*args, **kwargs):
    """
    Must be called before producer functions can be used.

    Merely a wrapper for calling kafka.KafkaProducer()
    """
    mykafka.init_producer(*args, **kwargs)

def produce_file(pub_names, key, thefile, hoststr, uname, tmpdir):
    """
    Synchronously produces `thefile` to each topic in `pub_names`.

    :param pub_names:   The names of the topics to produce to. Must be iterable.
    :param key:         The key in the Kafka key, value pair. Can be None, in which case,
                        a partition is chosen at random to write to.
    :param thefile:     The path to the file.
    :param hoststr:     The host URL like: nn1.example.com:50070
    :param uname:       The username.
    :param tmpdir:      The directory to store the temporary HDFS file.
    :returns:           None
    """
    fs = pyhdfs.HdfsClient(hosts=hoststr, user_name=uname)
    fname = os.path.basename(thefile)
    fs.create(tmpdir, fname)
    logging.debug("Producer requesting HDFS write " + str(hdfs_fpath))
    hdfs_fpath = tmpdir + "/" + fname
    mykafka.produce(pub_names, key, hdfs_fpath, serializer=lambda msg: msg.encode('utf8'))

def consume_file(con_names, hoststr, uname, delete_on_read=False):
    """
    Synchronously consumes raw file contents from files in each topic in `con_names` and
    yields them as a tuple of the form (fname, fcontents).

    :param con_names:       The names of the topics to consume from. Must be iterable.
    :param hoststr:         The host URL like: nn1.example.com:50070
    :param uname:           The username.
    :param delete_on_read:  If `True`, this function will delete each file from HDFS
                            after reading it.
    :returns:               Yields tuples of the form (fname, fcontents), one at a time.
    """
    fs = pyhdfs.HdfsClient(hosts=hoststr, user_name=uname)
    for hdfs_fpath in mykafka.consume(con_names, deserializer=lambda msg: msg.value.decode('utf8')):
        logging.debug("Consumer got HDFS file path: " + str(hdfs_fpath))
        with contextlib.closing(fs.open(hdfs_fpath)) as f:
            filecontents = f.read()
        if delete_on_read:
            fs.delete(hdfs_fpath)
        yield os.path.basename(hdfs_fpath), filecontents

def consume_and_produce(consume_names, callback, pub_names, hoststr, uname, tmpdir, delete_on_read=False):
    """
    Asynchronously consumes from topics `consume_names`, spawns a thread
    to deal with each message using `callback` and then takes `callback`'s
    result and produces it asynchronously to each of `pub_names`.

    :param consume_names:   The names of the topics to consume. Must be iterable.
    :param callback:        The function to apply on the item received
                            from consuming. This must take the file name and the
                            raw contents of
                            the file received and must return a tuple of the form
                            (new_fname, new_fcontents).
    :param pub_names:       The names of the topics to produce the transformed file contents to.
                            Must be iterable.
    :param hoststr:         The host URL like: nn1.example.com:50070
    :param uname:           The username.
    :param tmpdir:          The directory to store the temporary HDFS file.
    :param delete_on_read:  If `True`, this function will delete each file from HDFS
                            after reading it.
    :returns:               None
    """
    def callbackwrapper(hdfs_path):
        """
        Gets the file contents out of hdfs_path, applies `callback` to them
        and then puts the resulting file back into HDFS before returning
        the new HDFS file path.
        """
        # Get an HDFS client
        fs = pyhdfs.HdfsClient(hosts=hoststr, user_name=uname)

        # Read the file found at hdfs_path
        logging.debug("Attempting to read file contents from " + str(hdfs_path))
        with contextlib.closing(fs.open(hdfs_path)) as f:
            fcontents = f.read()

        # Apply the callback function to the contents of the file
        newfname, result_contents = callback(os.path.basename(hdfs_path), fcontents)

        # Maybe delete the file in HDFS
        if delete_on_read:
            fs.delete(hdfs_path)

        # Write the new contents to a temporary file
        # Then create a file in HDFS from that file
        with tempfile.TemporaryFile('w') as tf:
            tf.write(result_contents)
            fs.create(tmpdir, newfname)

    deserializer = lambda msg: msg.value.decode('utf8')
    serializer = lambda msg_payload: msg_payload.encode('utf8')
    mykafka.consume_and_produce(consume_names, deserializer, callbackwrapper, pub_names, serializer=serializer)

