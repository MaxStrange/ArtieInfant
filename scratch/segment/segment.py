"""
This module simply exposes a wrapper of a pydub.AudioSegment object.
"""
from __future__ import division

import collections
import pydub
import subprocess
import tempfile

class Segment:
    """
    This class is a wrapper for a pydubsegment. But it also adds several methods
    on top of it that I have found useful.
    """

    def __init__(self, pydubseg, name):
        self.seg = pydubseg
        self.name = name

    def __getattr__(self, attr):
        orig_attr = self.seg.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                if result == self.seg:
                    return self
                elif type(result) == pydub.AudioSegment:
                    return Segment(result, self.name)
                else:
                    return  result
            return hooked
        else:
            return orig_attr

    def __len__(self):
        return len(self.seg)

    def __eq__(self, other):
        return self.seg == other

    def __ne__(self, other):
        return self.seg != other

    def __iter__(self, other):
        return (x for x in self.seg)

    def __getitem__(self, millisecond):
        return Segment(self.seg[millisecond], self.name)

    def __add__(self, arg):
        if type(arg) == Segment:
            self.seg._data = self.seg._data + arg.seg._data
        else:
            self.seg = self.seg + arg
        return self

    def __radd__(self, rarg):
        return self.seg.__radd__(rarg)

    def __sub__(self, arg):
        if type(arg) == Segment:
            self.seg = self.seg - arg.seg
        else:
            self.seg = self.seg - arg
        return self

    def __mul__(self, arg):
        if type(arg) == Segment:
            self.seg = self.seg * arg.seg
        else:
            self.seg = self.seg * arg
        return self

    def detect_voice(self):
        """
        Returns self as a list of tuples:
        [('v', voiced segment), ('u', unvoiced segment), (etc.)]

        The overall order of the Segment is preserved.

        :returns: The described list. Does not modify self.
        """
        def vad_collector(frame_duration_ms, padding_duration_ms, v, frames):
            """
            Collects self into segments of VAD and non VAD.

            Yields tuples, one at a time, either ('v', Segment) or ('u', Segment).
            """
            construct_segment = lambda frames: Segment(pydub.AudioSegment(data=b''.join([f.bytes for f in frames]),
                                                                          sample_width=self.sample_width,
                                                                          frame_rate=self.frame_rate,
                                                                          channels=self.channels), self.name)
            def construct_segment(frames):
                return Segment(pydub.AudioSegment(data=b''.join([f.bytes
            num_padding_frames = int(padding_duration_ms / frame_duration_ms)
            ring_buffer = collections.deque(maxlen=num_padding_frames)
            triggered = False
            voiced_frames = []
            for frame in frames:
                if not triggered:
                    ring_buffer.append(frame)
                    num_voiced = len([f for f in ring_buffer if v.is_speech(f.bytes, sample_rate)])
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    ring_buffer.append(frame)
                    num_unvoiced = len([f for f in ring_buffer if not v.is_speech(f.bytes, sample_rate)])
                    if num_unvoiced > 0.9 * ring_buffer.maxlen:
                        triggered = False
                        yield 'v', construct_segment(voiced_frames)
                        yield 'u', construct_segment(ring_buffer)
                        ring_buffer.clear()
                        voiced_frames = []
            if voiced_frames:
                yield 'v', construct_segment(voiced_frames)
            if ring_buffer:
                yield 'u', construct_segment(ring_buffer)

        aggressiveness = 1
        window_size = 20
        padding_duration_ms = 800

        frames = self.generate_frames(frame_duration_ms=window_size, zero_pad=True)
        v = webrtcvad.Vad(int(aggressiveness))
        return [tup for tup in vad_collector(window_size, padding_duration_ms, v, frames)]

    def filter_silence(self):
        """
        Removes all silence from this segment and returns itself after modification.

        :returns: self, for convenience (self is modified in place as well)
        """
        tmp = tempfile.NamedTemporaryFile()
        othertmp = tempfile.NamedTemporaryFile()
        self.export(tmp, format="WAV")
        command = "sox " + tmp.name + " " + othertmp.name + " silence 1 0.8 0.1% reverse silence 1 0.8 0.1% reverse"
        res = subprocess.run(command.split(' '), stdout=subprocess.PIPE)
        assert proc.returncode == 0, "Sox did not work as intended, or perhaps you don't have Sox installed?"
        other = Segment(pydub.AudioSegment.from_wav(othertmp.name), self.name)
        tmp.close()
        othertmp.close()
        self = other
        return self

    def generate_frames(frame_duration_ms, zero_pad=True):
        """
        Yields self's data in chunks of frame_duration_ms.

        This function adapted from pywebrtc's example [https://github.com/wiseman/py-webrtcvad/blob/master/example.py].

        :param frame_duration_ms: The length of each frame in ms.
        :param zero_pad: Whether or not to zero pad the end of the Segment object to get all the audio data out as frames. If not,
                         there may be a part at the end of the Segment that is cut off (the part will be <= frame_duration_ms in length).
        :returns: A Frame object with properties 'bytes (the data)', 'timestamp (start time)', and 'duration'.
        """
        Frame = collections.namedtuple("Frame", "bytes timestamp duration")

        bytes_per_frame = int(self.sample_rate * (frame_duration_ms / 1000) * self.sample_width)  # (samples/sec) * (seconds in a frame) * (bytes/sample)
        offset = 0  # where we are so far in self's data (in bytes)
        timestamp = 0.0  # where we are so far in self (in seconds)
        frame_duration_s = (bytes_per_frame / self.sample_rate) / self.sample_width  # (bytes/frame) * (sample/bytes) * (sec/samples)
        while offset + bytes_per_frame < len(self.raw_data):
            yield Frame(self.raw_data[offset:offset + bytes_per_frame], timestamp, frame_duration_s
            timestamp += frame_duration_s
            offset += bytes_per_frame

        if zero_pad:
            rest = self.raw_data[offset:]
            zeros = bytes(bytes_per_frame - len(rest))
            yield Frame(rest + zeros, timestamp, frame_duration_s

    def reduce(self, others):
        """
        Reduces others into this one by concatenating all the others onto this one.

        :param others: The other Segment objects to append to this one.
        :returns: self, for convenience (self is modified in place as well)
        """
        for other in others:
            self.seg._data += other.seg._data

        return self

    def trim_to_minutes(self, strip_last_seconds=False):
        """
        Does not modify self, but instead returns a list of minute-long (at most) Segment objects.

        :param strip_last_seconds: If True, this method will return minute-long segments, but the last three seconds of this Segment won't be returned.
                                   This is useful for removing the microphone artifact at the end of the recording.
        :returns: A list of Segment objects, each of which is one minute long at most (and only the last one - if any - will be less than one minute).
        """
        starts = range(0, int(round(self.duration_seconds * MS_PER_S)), MS_PER_MIN)
        stops = (min(self.duration_seconds * MS_PER_S, start + MS_PER_MIN) for start in starts)
        wav_outs = [self[start:stop] for start, stop in zip(starts, stops)]

        # Now cut out the last three seconds of the last item in wav_outs (it will just be microphone artifact)
        # or, if the last item is less than three seconds, just get rid of it
        if strip_last_seconds:
            if wav_outs[-1].duration_seconds > 3:
                wav_outs[-1] = wav_outs[-1][:-MS_PER_S * 3]
            else:
                wav_outs = wav_outs[:-1]

        return wav_outs

