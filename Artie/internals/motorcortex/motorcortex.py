"""
This module contains code for controlling the articulatory synthesizer at a high level.
"""
import numpy as np
import output.voice.synthesizer as synth  # pylint: disable=locally-disabled, import-error
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
        self._allowed_values = {}
        for k, v in allowedvals:
            new_k = k.lower()
            new_v = config.make_list_from_string(v, itemtype=float)
            self._allowed_values[new_k] = new_v

        self._nagents_phase0 = config.getint('synthesizer', 'nagents-phase0')
        self._articulation_time_points_ms = config.getlist('synthesizer', 'articulation-time-points-ms', type=float)
        self._narticulators = len(self._allowed_values.keys())
        self._articulation_duration_ms = config.getfloat('synthesizer', 'phoneme-durations-ms')

    def pretrain(self):
        """
        Pretrains the model to make noise as loudly as possible.
        """
        # The shape of each agent is a flattend synthmat
        agentshape = (self._narticulators * len(self._articulation_time_points_ms), )

        # Create the fitness function
        fitnessfunction = ParallelizableFitnessFunctionPhase0(self._narticulators, self._articulation_duration_ms, self._articulation_time_points_ms)

        sim = po.Simulation(self._nagents_phase0, agentshape, fitnessfunction,
                            seedfunc=self._phase0_seed_function,
                            selectionfunc=self._phase0_selection_function,
                            crossoverfunc=self._phase0_crossover_function,
                            mutationfunc=self._phase0_mutation_function,
                            elitismfunc=None,
                            nworkers=self._nworkers,
                            max_agents_per_generation=self.population,
                            min_agents_per_generation=self.population)
        sim.run(niterations=self._phase0_niterations, fitness=self._phase0_fitness_target)

    def _phase0_seed_function(self):
        """
        Returns an agent of random uniform values between each articulator's min, max.
        """
        # TODO: return an agent
        pass

    def _phase0_selection_function(self, agents, fitnesses):
        """
        Take the top x percent.
        """
        # TODO: Return n agents
        pass

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
        # TODO: Return agents
        pass

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
        if seg.is_human_audible():
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
