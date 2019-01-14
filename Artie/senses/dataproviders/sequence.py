"""
Module for exposing Sequence implementation.
"""
import senses.dataproviders.featureprovider as fp
import logging
import multiprocessing
import os
import psutil

class Sequence():
    """
    Sequence of data, but buffered by parallel workers, so the queue that we
    yield from is hopefully never empty.
    """
    def __init__(self, ms_of_dataset, ms_per_batch, nworkers, root, sample_rate_hz, nchannels, bytewidth, provider_fun, *args, **kwargs):
        """
        :param ms_of_dataset:  The total number of ms in the dataset
        :param ms_per_batch:   The number of ms per batch
        :param nworkers:       The number of worker processes we will spawn using multiprocessing
        :param root:           The root directory to walk looking for data
        :param sample_rate_hz: The sample rate in Hz to resample any audio into
        :param nchannels:      The number of audio channels to resample into
        :param bytewidth:      The sample width in bytes to resample audio into
        :param provider_fun:   A str, which should be the function name of one of the batch provider functions in FeatureProvider, such as
                               'generate_n_fft_batches'.
        :param args:           Passed into `provider_fun`.
        :param kwargs:         Passed into `provider_fun`.
        """
        self.ms_of_dataset  = ms_of_dataset
        self.ms_per_batch   = ms_per_batch
        self.nworkers       = nworkers
        self.root           = root
        self.sample_rate_hz = sample_rate_hz
        self.nchannels      = nchannels
        self.bytewidth      = bytewidth
        self.provider_fun   = provider_fun
        self.args           = args
        self.kwargs         = kwargs
        self.queue          = multiprocessing.Queue(maxsize=20500)  # This is just an arbitrary limit so that it doesn't grow forever if the GPU is slower than the providers
        self.workers        = []

        for i in range(nworkers):
            proc = multiprocessing.Process(target=self._run_worker, args=(i,), daemon=True)
            proc.start()
            self.workers.append(proc)

    def __len__(self):
        return self.ms_of_dataset / self.ms_per_batch

    def __next__(self):
        # Return a batch from the batch producer
        return self.queue.get()

    def _run_worker(self, worker_idx):
        """
        """
        provider = fp.FeatureProvider(self.root, sample_rate=self.sample_rate_hz, nchannels=self.nchannels, bytewidth=self.bytewidth, worker_index=worker_idx)
        fn = getattr(provider, self.provider_fun)
        for idx, batch in enumerate(fn(*self.args, **self.kwargs)):
            self.queue.put(batch)
            if idx % 1000 == 0:
                thisproc = psutil.Process(os.getpid())
                msg = "MEM USAGE FOR WORKER {}: {} MB".format(worker_idx, thisproc.memory_info().rss/1E8)
                spaces = " " * len(msg) * worker_idx
                logging.debug("{} {}".format(spaces, msg))
