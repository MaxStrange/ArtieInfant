"""
This module contains code for controlling the articulatory synthesizer at a high level.
"""
import collections
import experiment.configuration as configuration # pylint: disable=locally-disabled, import-error
import logging
import numpy as np
import os
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

        # Get parameters for all phases
        self._articulation_time_points_ms = config.getlist('synthesizer', 'articulation-time-points-ms', type=float)
        self._narticulators = len(self._allowed_values.keys())
        self._articulation_duration_ms = config.getfloat('synthesizer', 'phoneme-durations-ms')
        self._nworkers = config.getint('synthesizer', 'nworkers-phase0')

        # Get parameters for Phase 0
        self._nagents_phase0 = config.getint('synthesizer', 'nagents-phase0')
        self._phase0_niterations = config.getstr('synthesizer', 'niterations-phase0')
        self._phase0_fitness_target = config.getstr('synthesizer', 'fitness-target-phase0')
        self._fraction_top_selection_phase0 = config.getfloat('synthesizer', 'fraction-of-generation-to-select-phase0')
        self._fraction_mutate_phase0 = config.getfloat('synthesizer', 'fraction-of-generation-to-mutate-phase0')
        self._anneal_after_phase0 = config.getbool('synthesizer', 'anneal-after-phase0')

        # Get parameters for Phase 1
        self._nagents_phase1 = config.getint('synthesizer', 'nagents-phase1')
        self._phase1_niterations = config.getstr('synthesizer', 'niterations-phase1')
        self._phase1_fitness_target = config.getstr('synthesizer', 'fitness-target-phase1')
        self._fraction_top_selection_phase1 = config.getfloat('synthesizer', 'fraction-of-generation-to-select-phase1')
        self._fraction_mutate_phase1 = config.getfloat('synthesizer', 'fraction-of-generation-to-mutate-phase1')
        self._anneal_during_phase1 = config.getbool('synthesizer', 'anneal-during-phase1')

        # Validate the fractions
        if self._fraction_mutate_phase0 < 0.0 or self._fraction_mutate_phase0 > 1.0:
            raise configuration.ConfigError("Mutation fraction must be within 0.0 and 1.0, but is {}.".format(self._fraction_mutate_phase0))

        if self._fraction_top_selection_phase0 < 0.0 or self._fraction_top_selection_phase0 > 1.0:
            raise configuration.ConfigError("Selection fraction must be within 0.0 and 1.0, but is {}.".format(self._fraction_top_selection_phase0))

        if self._fraction_mutate_phase1 < 0.0 or self._fraction_mutate_phase1 > 1.0:
            raise configuration.ConfigError("Mutation fraction must be within 0.0 and 1.0, but is {}.".format(self._fraction_mutate_phase1))

        if self._fraction_top_selection_phase1 < 0.0 or self._fraction_top_selection_phase1 > 1.0:
            raise configuration.ConfigError("Selection fraction must be within 0.0 and 1.0, but is {}.".format(self._fraction_top_selection_phase1))

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

        # Validate the phase 1 targets
        if self._phase1_niterations.lower().strip() == "none":
            self._phase1_niterations = None
        else:
            try:
                self._phase1_niterations = int(self._phase1_niterations)
            except ValueError:
                raise configuration.ConfigError("Cannot convert {} into an int. This value must be 'None' or an integer.".format(self._phase1_niterations))

        if self._phase1_fitness_target.lower().strip() == "none":
            self._phase1_fitness_target = None
        else:
            try:
                self._phase1_fitness_target = int(self._phase1_fitness_target)
            except ValueError:
                raise configuration.ConfigError("Cannot convert {} into an int. This value must be 'None' or an integer.".format(self._phase1_niterations))

        if self._phase0_niterations is None and self._phase0_fitness_target is None:
            raise configuration.ConfigError("niterations-phase0 and fitness-target-phase0 cannot both be None.")

        if self._phase1_niterations is None and self._phase1_fitness_target is None:
            raise configuration.ConfigError("niterations-phase1 and fitness-target-phase1 cannot both be None.")

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

        # We will be saving the populations sometimes
        self._phase0_population = None
        self._population_index = 0
        self.best_agents_phase0 = None
        self.best_agents_phase1 = None

    def _zero_limits(self, articulator_mask):
        """
        Returns self._allowed_lows and self._allowed_highs, but
        with each value zeroed in it if it is NOT part of `articulator_mask`.
        """
        # Make copies
        lows = np.copy(self._allowed_lows)
        highs = np.copy(self._allowed_highs)

        # Reshape into matrix form (narticulators, ntimepoints)
        lows = np.reshape(lows, (self._narticulators, -1))
        highs = np.reshape(highs, (self._narticulators, -1))

        # Make a zero version
        zero_lows = np.zeros_like(lows)
        zero_highs = np.zeros_like(highs)

        # Add back in the articulators of interest
        zero_lows[articulator_mask, :] = lows[articulator_mask, :]
        zero_highs[articulator_mask, :] = highs[articulator_mask, :]

        # Reshape back into vector form and return
        return np.reshape(zero_lows, (-1,)), np.reshape(zero_highs, (-1,))

    def pretrain(self):
        """
        Pretrains the model to make noise as loudly as possible.
        """
        # Create the fitness function
        fitnessfunction = ParallelizableFitnessFunctionPhase0(self._narticulators, self._articulation_duration_ms, self._articulation_time_points_ms)

        # Zero out the articulators we aren't using during phase 0 (but save the old limits)
        saved_lows = np.copy(self._allowed_lows)
        saved_highs = np.copy(self._allowed_highs)
        self._allowed_lows, self._allowed_highs = self._zero_limits(synth.laryngeal_articulator_mask)

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
        self.best_agents_phase0 = list(sim.best_agents)

        self._summarize_results(best, value, sim, "Phase0OutputSound.wav")

        # Save the population, since we will use this population as the seed for the next phase
        self._phase0_population = np.copy(sim._agents)

        # Restore the original lows and highs
        self._allowed_lows = saved_lows
        self._allowed_highs = saved_highs

        # If we want to anneal after phase 0, now's the time to do it
        if self._anneal_after_phase0:
            ## Reshape best agent into matrix form (narticulators, ntimepoints)
            bestmatrix = np.reshape(best, (self._narticulators, -1))

            ## Add/Subtract from each of its values
            bestmatrixlows = bestmatrix - 0.05
            bestmatrixhighs = bestmatrix + 0.05

            ## For each item in bestmatrixlows/highs, take the appropriate of min/max(best-lows/highs, currentlimits)
            ## This is so that we don't accidentally make the limits *less* stringent
            lows = np.reshape(self._allowed_lows, (self._narticulators, -1))
            highs = np.reshape(self._allowed_highs, (self._narticulators, -1))
            bestmatrixlows = np.maximum(bestmatrixlows, lows)
            bestmatrixhighs = np.minimum(bestmatrixhighs, highs)

            ## Now update our allowed highs/lows with the annealed values
            lows[synth.laryngeal_articulator_mask, :] = bestmatrixlows[synth.laryngeal_articulator_mask, :]
            highs[synth.laryngeal_articulator_mask, :] = bestmatrixhighs[synth.laryngeal_articulator_mask, :]
            self._allowed_lows = np.reshape(lows, (-1,))
            self._allowed_highs = np.reshape(highs, (-1,))

    def _run_phase1_simulation(self, target, niterations, fitness_target, savefpath):
        # Create the fitness function for phase 1
        fitnessfunction = ParallelizableFitnessFunctionPhase1(self._narticulators, self._articulation_duration_ms, self._articulation_time_points_ms, target)

        # Create the simulation and run it
        sim = po.Simulation(self._nagents_phase1, self._agentshape, fitnessfunction,
                            seedfunc=self._phase1_seed_function,
                            selectionfunc=self._phase1_selection_function,
                            crossoverfunc=None,  # Use default 2-point crossover function from library
                            mutationfunc=self._phase1_mutation_function,
                            elitismfunc=None,
                            nworkers=self._nworkers,
                            max_agents_per_generation=self._nagents_phase1,
                            min_agents_per_generation=self._nagents_phase1)
        best, value = sim.run(niterations=niterations, fitness=fitness_target)
        self.best_agents_phase1 = list(sim.best_agents)

        self._summarize_results(best, value, sim, savefpath)

    def train(self, target, savefpath=None):
        """
        Trains the model to mimic the given `target`, which should be an AudioSegment.

        If `savefpath` is not None, we will save the sound that corresponds to the best agent at this location
        as a WAV file.
        """
        if self._anneal_during_phase1:
            masks_in_order = [
                synth.jaw_articulator_mask,
                synth.nasal_articulator_mask,
                synth.lingual_articulator_support_mask,
                synth.lingual_articulator_tongue_mask,
                synth.labial_articulator_mask
            ]
            # If we pretrained already, we should add the laryngeal group to the annealed list
            annealed_masks = synth.laryngeal_articulator_mask if self._phase0_population is not None else []

            for maskidx, mask in enumerate(masks_in_order):
                # Backup the limits
                lows = np.copy(self._allowed_lows)
                highs = np.copy(self._allowed_highs)

                # Zero out the limits except for any that have already been annealed
                zeromask = sorted(list(set(annealed_masks + mask)))
                self._allowed_lows, self._allowed_highs = self._zero_limits(zeromask)

                # Our target for the simulation is based on which group we are training
                try:
                    niterations = self._phase1_niterations[maskidx]
                except TypeError:
                    niterations = self._phase1_niterations

                try:
                    fitnesstarget = self._phase1_fitness_target[maskidx]
                except TypeError:
                    fitnesstarget = self._phase1_fitness_target

                # Now run the simulation normally
                if savefpath is not None:
                    savefpath = os.path.splitext(savefpath)[0] + "_" + str(maskidx) + ".wav"
                self._run_phase1_simulation(target, niterations, fitnesstarget, savefpath)

                # Add this latest mask to the list of masks that we should anneal
                annealed_masks.extend(mask)

                # Restore the lows and highs
                self._allowed_lows = lows
                self._allowed_highs = highs
        else:
            # Run a normal simulation
            self._run_phase1_simulation(target, self._phase1_niterations, self._phase1_fitness_target, savefpath)

    def _summarize_results(self, best, value, sim, soundfpath):
        """
        Summarize `best` agent and `value`, which is its fitness.
        """
        # Reshape the agent into a synthesis matrix
        synthmat = np.reshape(best, (self._narticulators, len(self._articulation_time_points_ms)))

        # Print the synthmat in an easily-digestible format
        df = pandas.DataFrame(synthmat, index=synth.articularizers, columns=self._articulation_time_points_ms)
        logging.info("Best Value: {}; Agent:\n{}".format(value, df))

        if soundfpath:
            # Make a sound from this agent and save it for human consumption
            seg = synth.make_seg_from_synthmat(synthmat, self._articulation_duration_ms / 1000.0, [tp / 1000.0 for tp in self._articulation_time_points_ms])
            seg.export(soundfpath, format="WAV")
            sim.dump_history_csv(os.path.splitext(soundfpath)[0] + ".csv")

    def _phase0_seed_function(self):
        """
        Returns an agent of random uniform values between each articulator's min, max.
        """
        return np.random.uniform(self._allowed_lows, self._allowed_highs)

    def _phase1_seed_function(self):
        """
        If we have pretrained (done phase 0), we use the population from that phase, with some Gaussian noise
        added to them.

        If we do not have a pretrained population, we simply do a random uniform.
        """
        if self._phase0_population is not None:
            # Grab the next agent
            agent = self._phase0_population[self._population_index, :]

            # Adjust index for next time
            self._population_index += 1
            if self._population_index >= self._phase0_population.shape[0]:
                self._population_index = 0

            # Add some noise
            agent = np.random.normal(agent, 0.08)
            agent = np.clip(agent, self._allowed_lows, self._allowed_highs)

            return agent
        else:
            return np.random.uniform(self._allowed_lows, self._allowed_highs)

    def _phase0_selection_function(self, agents, fitnesses):
        """
        Take the top x percent.
        """
        nagents = int(agents.shape[0] * self._fraction_top_selection_phase0)
        if nagents < 1:
            nagents = 1

        return agents[0:nagents]

    def _phase1_selection_function(self, agents, fitnesses):
        """
        Take the top x percent.
        """
        nagents = int(agents.shape[0] * self._fraction_top_selection_phase1)
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
        agents[idxs, :] = np.random.normal(agents[idxs, :], 0.15)

        # make sure to clip to the allowed boundaries
        agents[idxs, :] = np.clip(agents[idxs, :], self._allowed_lows, self._allowed_highs)
        return agents

    def _phase1_mutation_function(self, agents):
        """
        Does exactly the same thing as Phase 0 to start with. If we need to change, we can.
        """
        return self._phase0_mutation_function(agents)

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
        # TODO: Human audible is rather loosely defined by an equation like this:
        # y = 40.11453 - 0.01683607x + 1.406211e-6x^2 - 2.371512e-11x^3
        # Where the Y axis is dB SPL and the X axis is frequency. Above this curve is audible, below it is not.
        # See https://www.etymotic.com/media/publications/erl-0096-1997.pdf (Hearing Thresholds by Yost and Killion, 1997)
        # Fitness might be something like: SPL of sound minus threshold value evaluated at the sound's characteristic frequency.
        if True:#seg.is_human_audible():
            return seg.rms
        else:
            return 0.0

class ParallelizableFitnessFunctionPhase1:
    def __init__(self, narticulators, duration_ms, time_points_ms, prototype_sound):
        """
        :param narticulators: How many articulators?
        :param duration_ms: The total ms of articulation we should create from each agent.
        :param time_points_ms: The time points (in ms) at which to change the values of each articulator.
        :param prototype_sound: AudioSegment of the prototypical sound for this index.
        """
        self.narticulators = narticulators
        self.duration_ms = duration_ms
        self.time_points_ms = time_points_ms
        self.ntimepoints = len(time_points_ms)

        # Forward process the target sound so that we don't have to do it every single time we execute
        target = prototype_sound.to_numpy_array().astype(float)
        target += abs(min(target))
        if max(target) != min(target):
            target /= max(target) - min(target)
        self._normalized_target = target
        assert sum(self._normalized_target[self._normalized_target < 0]) == 0

    def __call__(self, agent):
        """
        This fitness function evaluates an agent on how well it matches the prototype sound.
        """
        synthmat = np.reshape(agent, (self.narticulators, self.ntimepoints))
        seg = synth.make_seg_from_synthmat(synthmat, self.duration_ms / 1000.0, [tp / 1000.0 for tp in self.time_points_ms])

        # Shift the wave form up by most negative value
        ours = seg.to_numpy_array().astype(float)
        most_neg_val = min(ours)
        ours += abs(most_neg_val)
        if max(ours) != min(ours):
            ours /= max(ours) - min(ours)

        assert sum(ours[ours < 0]) == 0

        # Cross correlate with some amount of zero extension
        xcor = np.correlate(ours, self._normalized_target, mode='full')

        # Find the single maximum value along the xcor vector
        # This is the place at which the waves match each other best
        # Take the xcor value at this location as the reward
        # But also make sure the sound doesn't become inaudible
        rew = (0.8 * max(xcor)) + (0.2 * seg.rms)

        return rew
