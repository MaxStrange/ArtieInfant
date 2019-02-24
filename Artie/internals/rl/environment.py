"""
This module provides the observations and rewards for testing and
for training the RL agent to vocalize.
"""
import collections
import logging
import math
import numpy as np
import random
import output.voice.synthesizer as synth  # pylint: disable=locally-disabled, import-error
import warnings

Step = collections.namedtuple("Step", "state action")
ObservationSpace = collections.namedtuple("ObservationSpace", "dtype high low shape")

class TestEnvironment:
    """
    The test environment.
    """
    def __init__(self, behavior, nsteps, first_obs, action_shape, observation_space):
        """
        :param behavior:            A callable of signature fn(obs, action) -> (obs, reward, done).
                                    This function is what our step() function actually calls under the hood.
        :param nsteps:              The number of steps before we return 'done' for step(). If this parameter
                                    is None, an episode will only terminate if the behavior yields a done.
        :param first_obs:           The first observation that should be returned by calling reset(). This may be a callable.
        :param action_shape:        The shape of an action in this environment.
        :param observation_space:   An ObservationSpace.
        """
        self.behavior = behavior
        self.nsteps = nsteps
        self.nsteps_so_far_taken = 0
        self.first_obs = first_obs
        self.most_recent_obs = first_obs
        self.action_shape = action_shape
        self.observation_space = observation_space

    def reset(self):
        """
        Reset the environment and give the first observation.

        :returns:           The first observation of the environment.
        """
        self.nsteps_so_far_taken = 0

        if callable(self.first_obs):
            obs = self.first_obs()
        else:
            obs = self.first_obs
        self.most_recent_obs = obs
        return obs

    def step(self, action):
        """
        :param action:      The action to take at the current step to derive the next
        :returns:           ob, rew, done, info; where:
                            - ob:   The observation of the step that we are at in response to
                                    taking `action`.
                            - rew:  The reward we got for taking `action`.
                            - done: True or False, depending on if the episode is over.
                            - info: Dict of random crap. Currently always empty.
        """
        if self.nsteps is not None and self.nsteps_so_far_taken >= self.nsteps:
            # Terminate the episode
            self.nsteps_so_far_taken = 0
            return self.most_recent_obs, 0, True, {}
        else:
            # Increment the number of steps
            self.nsteps_so_far_taken += 1

            # Figure out our next step
            obs, reward, done = self.behavior(self.most_recent_obs, action, self.most_recent_obs)

            # update the most recent observation
            self.most_recent_obs = obs

            # Return the observation, reward, whether we are done or not, and the info dict
            return obs, reward, done, {}

class SomEnvironment:
    """
    This class provides the Environment for learning to vocalize and to produce sounds that are 'phoneme'-like.

    The behavior of this environment is as follows. Each episode is exactly one step. The first (and only)
    observation that is given is a random (uniform) scalar that represents the cluster index of a sound,
    as clustered by: Sound -> Preprocessing -> VAE -> Mean Shift Clustering over all latent vectors produced during VAE training.
    The action space is len(articularizers) (continuous). The reward function depends on if this environment
    is in phase 0 or phase 1 of training. During phase 0, a reward is given based on whether or not an
    audible sound is produced via the chosen action, as fed into the articulatory synthesizer controller.
    During phase 1, the reward is conditioned on the resulting sound sounding like the prototype of the
    cluster observed, probably via a correlation DSP function.
    """
    def __init__(self, nclusters, articulation_duration_ms, time_points_ms, cluster_prototypes):
        """
        :param nclusters:                   The number of clusters.
        :param articulation_duration_ms:    The total number of ms of each articulation. Currently we only support
                                            making all articulations the same duration.
        :param time_points_ms:              The discrete time points of the articulation. This parameter
                                            indicates the number of times we will move the articularizers and when.
                                            The action space is of shape (narticularizers, ntime_points).
        :param cluster_prototypes:          A list or dict of the form [cluster_index => cluster_prototype]. Each cluster prototype
                                            should be an AudioSegment object.
        """
        for idx, tp in enumerate(time_points_ms):
            if tp < 0:
                raise ValueError("Time points cannot be negative. Got", tp, "at index", idx, "in 'time_points' parameter.")
            elif tp > articulation_duration_ms:
                raise ValueError("Time point", tp, "at index", idx, "is greater than the duration of the articulation:", articulation_duration_ms)

        self.nclusters = nclusters
        self._phase = 0
        self._retain_audio = False
        self._audio_buffer = []
        self.observed_cluster_index = None
        self.articulation_duration_ms = articulation_duration_ms
        self.time_points_ms = time_points_ms
        self.cluster_prototypes = cluster_prototypes
        self.action_shape = (len(synth.articularizers), len(time_points_ms))
        self.observation_space = ObservationSpace(dtype=np.int32,
                                                  high=np.array([(self.nclusters - 1)], dtype=np.int32),
                                                  low=np.array([0], dtype=np.int32),
                                                  shape=(1,))
        self._inference_mode = False
        self._previous_inferred_index = -1
        self._xcor_component_max = 0.0
        self._step = 0

    @property
    def inference_mode(self):
        """Inference mode = True means that we cycle through the observations rather than sampling them randomly"""
        return self._inference_mode

    @inference_mode.setter
    def inference_mode(self, v):
        """Inference mode = True means that we cycle through the observations rather than sampling them randomly"""
        self._inference_mode = v

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        """Set the phase. Phase is zero or one. Any nonzero value will set phase to one."""
        if p != 0:
            self._phase = 1
        else:
            self._phase = 0

    @property
    def retain_audio(self):
        """
        Set to True if you want to keep the audio that is generated by the Agent.
        The audio will be retained in self.produced_audio. You may dump the audio to disk
        with `env.dump_audio()` or you may clear the audio buffer with `env.clear_audio()`.
        """
        return self._retain_audio

    @retain_audio.setter
    def retain_audio(self, retain):
        self._retain_audio = retain

    def clear_audio(self):
        """
        Clear the audio buffer. The audio buffer is the audio generated by the agent if
        this environment's `retain_audio` is set to True.
        """
        self._audio_buffer.clear()

    def dump_audio(self, basefname=None):
        """
        Dumps each audio segment from the audio buffer to disk. Does not clear the buffer.

        :param basefname:   If not None, the base filename, which will have the audio segment
                            indexes appended starting from 0. Can be a file path. If None,
                            the default name of 'produced_sound' is used as the base.
        """
        base = basefname if basefname is not None else "produced_sound"
        for i, seg in enumerate(self._audio_buffer):
            seg.export("{}{}.wav".format(base, i), format='wav')

    def reset(self):
        """
        Reset the environment and give the first observation.
        Will NOT clear the audio buffer as well.

        :returns:           The first observation of the environment, a uniform random scalar
                            from the distribution [0, self.nclusters]. If we are in inference mode
                            however, we will return 0 first, then 1, ..., self.nclusters - 1, then 0, etc.
        """
        if self.inference_mode:
            self.observed_cluster_index = (self._previous_inferred_index + 1) % self.nclusters
        else:
            self.observed_cluster_index = random.choice([n for n in range(0, self.nclusters)])

        self._previous_inferred_index = self.observed_cluster_index
        return np.array([self.observed_cluster_index], dtype=self.observation_space.dtype)

    def step(self, action):
        """
        :param action:      The action to take at the current step to derive the next
        :returns:           ob, rew, done, info; where:
                            - ob:   The observation of the step that we are at in response to
                                    taking `action`.
                            - rew:  The reward we got for taking `action`.
                            - done: True or False, depending on if the episode is over.
                            - info: Dict of random crap. Currently always empty.
        """
        action = np.reshape(action, self.action_shape)

        # Just return the cluster index we generated at reset as the observation
        obs  = np.array([self.observed_cluster_index], dtype=self.observation_space.dtype)
        done = True # We are always done after the first step in this environment
        info = {}   # Info dict is just an empty dict, kept for compliance with OpenAI Gym

        warnings.simplefilter(action='ignore', category=ResourceWarning)
        seg = synth.make_seg_from_synthmat(action, self.articulation_duration_ms / 1000.0, [tp / 1000.0 for tp in self.time_points_ms])
        if self.retain_audio:
            self._audio_buffer.append(seg)

        ##################### Used to Debug. Should not be retained while actually using this ########################
        if self._step % 50 == 0:
            seg.export("mimicry-{}-{}-debug-delete-me.wav".format(self.observed_cluster_index, self._step), format='wav')
        ##############################################################################################################

        if self.phase == 0:
            # During phase 0, the reward is based on whether or not we vocalized at all
            arr = seg.to_numpy_array()
            assert len(arr) > 0
            squares = np.square(arr)
            assert len(squares) == len(arr)
            sum_of_squares = np.sum(squares[squares >= 0], axis=0)
            assert sum_of_squares >= 0.0, "Len: {}, Sum of squares: {}".format(len(arr), np.sum(squares, axis=0))
            mean_square = sum_of_squares / len(arr)
            assert mean_square > 0.0
            rms = np.sqrt(mean_square)
            rew = rms
            if math.isnan(rew):
                rew = 0.0
            rew /= 100.0  # Get it into a more reasonable domain
        else:
            # During phase 1, the reward is based on how well we match the prototype sound
            # for the given cluster index

            # Shift the wave form up by most negative value
            ours = seg.to_numpy_array().astype(float)
            most_neg_val = min(ours)
            ours += abs(most_neg_val)

            prototype = self.cluster_prototypes[int(self.observed_cluster_index)].to_numpy_array().astype(float)
            most_neg_val = min(prototype)
            prototype += abs(most_neg_val)

            assert sum(ours[ours < 0]) == 0
            assert sum(prototype[prototype < 0]) == 0

            # Divide by the amplitude
            if max(ours) != min(ours):
                ours /= max(ours) - min(ours)
            if max(prototype) != min(prototype):
                prototype /= max(prototype) - min(prototype)

            # Now you have values in the interval [0, 1]

            # XCorr with some amount of zero extension
            xcor = np.correlate(ours, prototype, mode='full')

            # Find the single maximum value along the xcor vector
            # This is the place at which the waves match each other best
            # Take the xcor value at this location as the reward
            rew = max(xcor)

        logging.debug("Observation: {} -> Action: {} -> Reward: {}".format(obs, action, rew))
        self._step += 1
        return obs, rew, done, info
