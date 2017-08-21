"""
This module simply exposes a wrapper of a pydub.AudioSegment object.
"""
import pydub


class Segment:
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

    def reduce(self, others):
        """
        Reduces others into this one by concatenating all the others onto this one.

        Returns:
            self, for convenience (self is modified in place as well)
        """
        for other in others:
            self.seg._data += other.seg._data

        return self
