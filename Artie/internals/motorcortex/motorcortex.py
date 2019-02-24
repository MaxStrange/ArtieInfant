"""
This module contains code for controlling the articulatory synthesizer at a high level.
"""
import output.voice.synthesizer as synth  # pylint: disable=locally-disabled, import-error

class SynthModel:
    """
    This class provides methods for pretraining and training the underlying machine
    learning model that controls the articulatory synthesizer.
    """
    def __init__(self, config):
        """
        Takes a configuration object for its hyperparameters.
        """
        pass

    def pretrain(self):
        """
        Pretrains the model to make noise as loudly as possible.
        """
        # During phase 0, the reward is based on whether or not we vocalized at all
        # fitnessfunc = lambda seg: seg.rms

#warnings.simplefilter(action='ignore', category=ResourceWarning)
#seg = synth.make_seg_from_synthmat(action, self.articulation_duration_ms / 1000.0, [tp / 1000.0 for tp in self.time_points_ms])
#if self.retain_audio:
#    self._audio_buffer.append(seg)
#
#    # During phase 1, the reward is based on how well we match the prototype sound
#    # for the given cluster index
#
#    # Shift the wave form up by most negative value
#    ours = seg.to_numpy_array().astype(float)
#    most_neg_val = min(ours)
#    ours += abs(most_neg_val)
#
#    prototype = self.cluster_prototypes[int(self.observed_cluster_index)].to_numpy_array().astype(float)
#    most_neg_val = min(prototype)
#    prototype += abs(most_neg_val)
#
#    assert sum(ours[ours < 0]) == 0
#    assert sum(prototype[prototype < 0]) == 0
#
#    # Divide by the amplitude
#    if max(ours) != min(ours):
#        ours /= max(ours) - min(ours)
#    if max(prototype) != min(prototype):
#        prototype /= max(prototype) - min(prototype)
#
#    # Now you have values in the interval [0, 1]
#
#    # XCorr with some amount of zero extension
#    xcor = np.correlate(ours, prototype, mode='full')
#
#    # Find the single maximum value along the xcor vector
#    # This is the place at which the waves match each other best
#    # Take the xcor value at this location as the reward
#    rew = max(xcor)
#
