"""
This module contains code for controlling the articulatory synthesizer at a high level.
"""
import collections
import experiment.configuration as configuration # pylint: disable=locally-disabled, import-error
import numpy as np
import output.voice.synthesizer as synth  # pylint: disable=locally-disabled, import-error
import pandas
import primordialooze as po

class SynthModel:
    """
    This class provides methods for pretraining and training the underlying machine
    learning model that controls the articulatory synthesizer.
    """
    def __init__(self, config):
        """
        Takes a configuration object for its hyperparameters.
        """
        allowedvals = config.getdict('synthesizer', 'allowed-articulator-values')
        # We are given a dict of str: str, but we want a dict of str: list{float}.
        # So try to do the conversion, raising an appropriate exception if unsuccessful.
        self._allowed_values = collections.OrderedDict()
        for k, v in allowedvals.items():
            new_k = k.lower()
            new_v = config.make_list_from_str(v, type=float)
            self._allowed_values[new_k] = new_v

        self._nagents_phase0 = config.getint('synthesizer', 'nagents-phase0')
        self._articulation_time_points_ms = config.getlist('synthesizer', 'articulation-time-points-ms', type=float)
        self._narticulators = len(self._allowed_values.keys())
        self._articulation_duration_ms = config.getfloat('synthesizer', 'phoneme-durations-ms')
        self._nworkers = config.getint('synthesizer', 'nworkers-phase0')
        self._phase0_niterations = config.getstr('synthesizer', 'niterations-phase0')
        self._phase0_fitness_target = config.getstr('synthesizer', 'fitness-target-phase0')
        self._fraction_top_selection_phase0 = config.getfloat('synthesizer', 'fraction-of-generation-to-select-phase0')
        self._fraction_mutate_phase0 = config.getfloat('synthesizer', 'fraction-of-generation-to-mutate-phase0')

        # Validate the fractions
        if self._fraction_mutate_phase0 < 0.0 or self._fraction_mutate_phase0 > 1.0:
            raise configuration.ConfigError("Mutation fraction must be within 0.0 and 1.0, but is {}.".format(self._fraction_mutate_phase0))

        if self._fraction_top_selection_phase0 < 0.0 or self._fraction_top_selection_phase0 > 1.0:
            raise configuration.ConfigError("Selection fraction must be within 0.0 and 1.0, but is {}.".format(self._fraction_top_selection_phase0))

        # Validate the phase 0 targets
        if self._phase0_niterations.lower().strip() == "none":
            self._phase0_niterations = None
        else:
            try:
                self._phase0_niterations = int(self._phase0_niterations)
            except ValueError:
                raise configuration.ConfigError("Cannot convert {} into an int. This value must be 'None' or an integer.".format(self._phase0_niterations))

        if self._phase0_fitness_target.lower().strip() == "none":
            self._phase0_fitness_target = None
        else:
            try:
                self._phase0_fitness_target = int(self._phase0_fitness_target)
            except ValueError:
                raise configuration.ConfigError("Cannot convert {} into an int. This value must be 'None' or an integer.".format(self._phase0_niterations))

        if self._phase0_niterations is None and self._phase0_fitness_target is None:
            raise configuration.ConfigError("niterations-phase0 and fitness-target-phase0 cannot both be None.")

        # The shape of each agent is a flattend synthmat
        self._agentshape = (self._narticulators * len(self._articulation_time_points_ms), )

        # Create a lows array and a highs array from the ordered dict of allowedvalues
        # These arrays should be the same length as an agent (ntimepoints * narticulators)
        # and each value in the array should correspond to a min/max value allowed at that
        # timepoint.
        self._allowed_lows = np.zeros((self._narticulators, len(self._articulation_time_points_ms)))
        self._allowed_highs = np.zeros((self._narticulators, len(self._articulation_time_points_ms)))
        for i, (_, v) in enumerate(self._allowed_values.items()):
            # Each value in this dict is a list of floats. Every two of these floats should be
            # a min/max pair of allowable values in the time series.
            if len(v) != 2 * len(self._articulation_time_points_ms):
                raise configuration.ConfigError("Could not parse the allowable synth values matrix. One of the lists has {} items, but all of them must have {}.".format(len(v), 2 * len(self._articulation_time_points_ms)))

            mins = [value for j, value in enumerate(v) if j % 2 == 0]
            maxes = [value for j, value in enumerate(v) if j % 2 == 1]
            self._allowed_lows[i, :] = np.array(mins)
            self._allowed_highs[i, :] = np.array(maxes)

        self._allowed_lows = np.reshape(self._allowed_lows, self._agentshape)
        self._allowed_highs = np.reshape(self._allowed_highs, self._agentshape)

    def pretrain(self):
        """
        Pretrains the model to make noise as loudly as possible.
        """
        # Create the fitness function
        fitnessfunction = ParallelizableFitnessFunctionPhase0(self._narticulators, self._articulation_duration_ms, self._articulation_time_points_ms)

        sim = po.Simulation(self._nagents_phase0, self._agentshape, fitnessfunction,
                            seedfunc=self._phase0_seed_function,
                            selectionfunc=self._phase0_selection_function,
                            crossoverfunc=self._phase0_crossover_function,
                            mutationfunc=self._phase0_mutation_function,
                            elitismfunc=None,
                            nworkers=self._nworkers,
                            max_agents_per_generation=self._nagents_phase0,
                            min_agents_per_generation=self._nagents_phase0)
        best, value = sim.run(niterations=self._phase0_niterations, fitness=self._phase0_fitness_target)
        print("Best agent: {}. Value: {}".format(best, value))

        # TODO: Save this population, not just the best agent and value

        synthmat = np.reshape(best, (self._narticulators, len(self._articulation_time_points_ms)))

        # Print the synthmat in an easily-digestible format
        df = pandas.DataFrame(synthmat, index=synth.articularizers, columns=self._articulation_time_points_ms)
        print(df)

        # Make a sound from this agent and save it for human consumption
        seg = synth.make_seg_from_synthmat(synthmat, self._articulation_duration_ms / 1000.0, [tp / 1000.0 for tp in self._articulation_time_points_ms])
        seg.export("OutputSound.wav", format="WAV")
        exit()

    def _phase0_seed_function(self):
        """
        Returns an agent of random uniform values between each articulator's min, max.
        """
        return np.random.uniform(self._allowed_lows, self._allowed_highs)

    def _phase0_selection_function(self, agents, fitnesses):
        """
        Take the top x percent.
        """
        nagents = int(agents.shape[0] * self._fraction_top_selection_phase0)
        if nagents < 1:
            nagents = 1

        return agents[0:nagents]

    def _phase0_crossover_function(self, agents):
        """
        For now, does nothing. Genetic variation is introduced solely by mutation.
        """
        return agents

    def _phase0_mutation_function(self, agents):
        """
        Mutates some number of agents each generation via Gaussian distributions
        with mean=value to mutate, stdev=mutation_stdev.

        Ensures that the agents do not stray outside the allowed bounds for values.
        """
        nagents = int(self._fraction_mutate_phase0 * agents.shape[0])
        if nagents < 1:
            nagents = 1

        idxs = np.random.choice(agents.shape[0], size=nagents, replace=False)
        agents[idxs, :] = np.random.normal(agents[idxs, :], 0.25)

        # make sure to clip to the allowed boundaries
        agents[idxs, :] = np.clip(agents[idxs, :], self._allowed_lows, self._allowed_highs)
        return agents

class ParallelizableFitnessFunctionPhase0:
    def __init__(self, narticulators, duration_ms, time_points_ms):
        """
        :param narticulators: How many articulators?
        :param duration_ms: The total ms of articulation we should create from each agent.
        :param time_points_ms: The time points (in ms) at which to change the values of each articulator.
        """
        self.narticulators = narticulators
        self.duration_ms = duration_ms
        self.time_points_ms = time_points_ms
        self.ntimepoints = len(time_points_ms)

    def __call__(self, agent):
        """
        This fitness function evaluates an agent on how much sound it makes when run through
        the articulatory synthesizer as a synthmat.
        """
        synthmat = np.reshape(agent, (self.narticulators, self.ntimepoints))
        seg = synth.make_seg_from_synthmat(synthmat, self.duration_ms / 1000.0, [tp / 1000.0 for tp in self.time_points_ms])

        # The fitness of an agent in this phase is determined by the RMS of the sound it makes,
        # UNLESS it fails to make a human-audible sound. In which case, it is assigned a fitness of zero.
        # TODO
        if True:#seg.is_human_audible():
            return seg.rms
        else:
            return 0.0

class ParallelizableFitnessFunctionPhase1:
    def __init__(self, narticulators, duration_ms, time_points_ms, prototype_sound, prototype_index):
        """
        :param narticulators: How many articulators?
        :param duration_ms: The total ms of articulation we should create from each agent.
        :param time_points_ms: The time points (in ms) at which to change the values of each articulator.
        :param prototype_sound: AudioSegment of the prototypical sound for this index.
        :param prototype_index: The raw index for this proto-phoneme cluster.
        """
        self.narticulators = narticulators
        self.duration_ms = duration_ms
        self.time_points_ms = time_points_ms
        self.ntimepoints = len(time_points_ms)
        self.prototype_index = prototype_index

        # TODO: Calculate what you need from the sound

    def __call__(self, agent):
        """
        This fitness function evaluates an agent on how well it matches the prototype sound.
        """
        synthmat = np.reshape(agent, (self.narticulators, self.ntimepoints))
        seg = synth.make_seg_from_synthmat(synthmat, self.duration_ms / 1000.0, [tp / 1000.0 for tp in self.time_points_ms])

        # TODO:
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
