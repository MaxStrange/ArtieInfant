"""
Detect voice or not voice in each segment.
"""
import collections
import os
import segment
import webrtcvad

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def _frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def _vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, v, frames, segment):
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
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def vad(segments):
    """
    !! This function modifies the segments you give it !!

    Returns:
        [Segment, Segment, Segment, etc.] where each Segment object has had its name adjusted so that it says whether it is voice or not.
    """
    # Parameters
    aggressiveness = 1
    window_size = 20
    padding_duration_ms = 20

    result = []
    for seg in segments:
        Fs = seg.frame_rate
        X = seg.get_array_of_samples()
        v = webrtcvad.Vad(int(aggressiveness))
        frames = [f for f in _frame_generator(window_size, X, Fs)]
        X = [voiced for voiced in _vad_collector(Fs, window_size, padding_duration_ms, v, frames, seg)]
        name, _extension = os.path.splitext(seg.name)
        name += "_voiced.WAV" if X else "_unv.WAV"
        seg.name = name
        seg.raw_data = X
        result.append(seg)
    return result

