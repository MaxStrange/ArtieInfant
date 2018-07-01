"""
Module for exposing Sequence implementation.
"""
import senses.dataproviders.featureprovider as fp
import multiprocessing

class Sequence():
    """
    Sequence of data, but buffered by parallel workers, so the queue that we
    yield from is hopefully never empty.
    """
    def __init__(self, ms_of_dataset, ms_per_batch, nworkers, root, sample_rate_hz, nchannels, bytewidth, provider_fun, *args, **kwargs):
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
        self.queue          = multiprocessing.Queue()

        for i in range(nworkers):
            proc = multiprocessing.Process(target=self._run_worker, args=(i,), daemon=True)
            proc.start()

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
        for batch in fn(*self.args, **self.kwargs):
            self.queue.put(batch)
