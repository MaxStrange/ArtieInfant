"""
Test the reinforcement learning algorithm.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras

import audiosegment
import enum
import logging
import numpy as np
import os
import random
import sys
import unittest

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import internals.rl.agent as agent              # pylint: disable=locally-disabled, import-error
import internals.rl.environment as environment  # pylint: disable=locally-disabled, import-error

# Configure the logger
logging.basicConfig(filename="TestLog.log", filemode='w', level=logging.DEBUG)

class XORSingleStepTestRig:
    """
    XOR environment where each step is terminal and randomly drawn from the four possible inputs.
    """
    def __init__(self):
        pass

    @staticmethod
    def observation_space():
        dtype = np.float32
        high = np.array([1, 1], dtype=np.float32)
        low = np.array([0, 0], dtype=np.float32)
        shape = (2,)
        return environment.ObservationSpace(dtype, high, low, shape)

    @staticmethod
    def action_shape():
        return (1,)

    @staticmethod
    def sample():
        """
        Return one of the four possible observations with uniform random chance.
        """
        choices = [
            np.array([1, 1], dtype=np.float32),
            np.array([1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32),
            np.array([0, 0], dtype=np.float32)
        ]

        return random.choice(choices)

    @staticmethod
    def behavior(obs, action, lastobs):
        """
         Create an environment of the form:
        {
            Step.state of [1, 1]:
            Step.state of [0, 0]:
                Step.action == 0      => Reward = +1, end
                Step.action otherwise => Reward = -1, end

            Step.state of [1, 0]:
            Step.state of [0, 1]:
                Step.action == 1      => Reward = +1, end
                Step.action otherwise => Reward = -1, end
        }

        Every step is terminal, regardless of action chosen. Each observation
        is randomly drawn from the four possibilities at reset.
        """
        if not hasattr(action, 'shape'):
            action = np.array([action], dtype=np.float32)
        assert action.shape == (1,)

        obslist = [int(round(i)) for i in obs.tolist()]

        if obslist == [1, 1] or obslist == [0, 0]:
            if int(round(action[0])) == 0:
                o = lastobs
                r = 1
                d = True
            else:
                o = lastobs
                r = -1
                d = True
        else:
            if int(round(action[0])) == 1:
                o = lastobs
                r = 1
                d = True
            else:
                o = lastobs
                r = -1
                d = True

        logging.debug("Observation: {} -> Action: {} -> Reward: {}".format(o, int(round(action[0])), r))
        return o, r, d


class XORTestRig:
    """
    XOR environment that goes through all four possible XOR inputs in order.
    """
    def __init__(self):
        pass

    @staticmethod
    def observation_space():
        dtype = np.float32
        high = np.array([1, 1], dtype=np.float32)
        low = np.array([0, 0], dtype=np.float32)
        shape = (2,)
        return environment.ObservationSpace(dtype, high, low, shape)

    @staticmethod
    def action_shape():
        return (1,)

    @staticmethod
    def behavior(obs, action, lastobs):
        """
        Create an environment of the form:
        {
            Step.state of [1, 1]:
            Step.state of [0, 0]:
                Step.action == 0      => Reward = +1
                Step.action otherwise => Reward = -1, end

            Step.state of [1, 0]:
            Step.state of [0, 1]:
                Step.action == 1      => Reward = +1
                Step.action otherwise => Reward = -1, end
        }

        Environment proceeds through the list:
        [0, 0]
        [0, 1]
        [1, 0]
        [1, 1]
        and ends after the last item or as soon as the agent makes
        an incorrect response.
        """
        if not hasattr(action, 'shape'):
            action = np.array([action], dtype=np.float32)
        assert action.shape == (1,)

        obslist = [int(round(i)) for i in obs.tolist()]

        if obslist == [1, 1] or obslist == [0, 0]:
            if int(round(action[0])) == 0:
                nextobs = lastobs if obslist == [1, 1] else np.array([0, 1], dtype=np.float32)
                done = obslist == [1, 1]
                o = nextobs
                r = 1
                d = done
            else:
                o = lastobs
                r = -1
                d = True
        else:
            if int(round(action[0])) == 1:
                nextobs = np.array([1, 0], dtype=np.float32) if obslist == [0, 1] else np.array([1, 1], dtype=np.float32)
                o = nextobs
                r = 1
                d = False
            else:
                o = lastobs
                r = -1
                d = True

        logging.debug("Observation: {} -> Action: {} -> Reward: {}".format(o, int(round(action[0])), r))
        return o, r, d

class TestRL(unittest.TestCase):
    __skip_the_long_ones = False

    class FirstObs(enum.Enum):
        XOR = np.array([0, 0], dtype=np.float32)

    def setUp(self):
        pass

    def test_xor_manual(self):
        """
        Tests that the environment test harness is working properly.
        """
        env = environment.TestEnvironment(XORTestRig.behavior,
                                        nsteps=4,
                                        first_obs=TestRL.FirstObs.XOR.value,
                                        action_shape=XORTestRig.action_shape(),
                                        observation_space=XORTestRig.observation_space())
        obs00 = env.reset()

        # Assert that the first observation we get is what we loaded as the first observation
        self.assertListEqual(obs00.tolist(), TestRL.FirstObs.XOR.value.tolist())

        # Go through the happy path and make sure each step is expected
        # Step 1
        obs01, rew, done, _info = env.step(0)
        self.assertListEqual(obs01.tolist(), [0, 1])
        self.assertEqual(rew, 1)
        self.assertEqual(done, False)

        # Step 2
        obs10, rew, done, _info = env.step(1)
        self.assertListEqual(obs10.tolist(), [1, 0])
        self.assertEqual(rew, 1)
        self.assertEqual(done, False)

        # Step 3
        obs11, rew, done, _info = env.step(1)
        self.assertListEqual(obs11.tolist(), [1, 1])
        self.assertEqual(rew, 1)
        self.assertEqual(done, False)

        # Step 4
        obs, rew, done, _info = env.step(0)
        self.assertIsNotNone(obs)
        self.assertEqual(rew, 1)
        self.assertEqual(done, True)

        # Test reset
        obs00 = env.reset()
        self.assertListEqual(obs00.tolist(), TestRL.FirstObs.XOR.value.tolist())

        # Test incorrect at each step
        # Step 1
        obs, rew, done, _info = env.step(1)
        self.assertIsNotNone(obs)
        self.assertEqual(rew, -1)
        self.assertEqual(done, True)
        env.reset()

        # Step 2
        _ = env.step(0)
        obs, rew, done, _info = env.step(0)
        self.assertIsNotNone(obs)
        self.assertEqual(rew, -1)
        self.assertEqual(done, True)
        env.reset()

        # Step 3
        _ = env.step(0)
        _ = env.step(1)
        obs, rew, done, _info = env.step(0)
        self.assertIsNotNone(obs)
        self.assertEqual(rew, -1)
        self.assertEqual(done, True)
        env.reset()

        # Step 4
        _ = env.step(0)
        _ = env.step(1)
        _ = env.step(1)
        obs, rew, done, _info = env.step(1)
        self.assertIsNotNone(obs)
        self.assertEqual(rew, -1)
        self.assertEqual(done, True)

    def _xor_actor(self, env):
        """
        Construct the actor part of the model and return it.
        """
        nactions = np.product(env.action_shape)

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(keras.layers.Dense(16))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(8))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(nactions))
        model.add(keras.layers.Activation('sigmoid'))
        return model

    def _som_actor(self, env):
        """
        Construct the actor part of the model and return it.
        """
        nactions = np.product(env.action_shape)

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(keras.layers.Dense(400))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(200))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(nactions))
        model.add(keras.layers.Activation('sigmoid'))
        return model

    def _xor_critic(self, env):
        """
        Construct the critic part of the model and return it.
        """
        nactions      = np.product(env.action_shape)
        action_input  = keras.layers.Input(shape=(nactions,), name='action_input')
        obs_input     = keras.layers.Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_obs = keras.layers.Flatten()(obs_input)

        out = keras.layers.Concatenate()([action_input, flattened_obs])
        out = keras.layers.Dense(16)(out)
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.Dense(8)(out)
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.Dense(1)(out)  # Must be single output
        out = keras.layers.Activation('linear')(out)
        critic = keras.models.Model(inputs=[action_input, obs_input], outputs=out)
        return critic, action_input

    def _som_critic(self, env):
        """
        Construct the critic part of the model and return it.
        """
        nactions      = np.product(env.action_shape)
        action_input  = keras.layers.Input(shape=(nactions,), name='action_input')
        obs_input     = keras.layers.Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_obs = keras.layers.Flatten()(obs_input)

        out = keras.layers.Concatenate()([action_input, flattened_obs])
        out = keras.layers.Dense(400)(out)
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.Dense(200)(out)
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.Dense(1)(out)  # Must be single output
        out = keras.layers.Activation('linear')(out)
        critic = keras.models.Model(inputs=[action_input, obs_input], outputs=out)
        return critic, action_input

    def test_xor_via_agent(self):
        """
        Test constructing an Agent and then learning the XOR environment with it.
        """
        env = environment.TestEnvironment(XORTestRig.behavior,
                                        nsteps=4,
                                        first_obs=TestRL.FirstObs.XOR.value,
                                        action_shape=XORTestRig.action_shape(),
                                        observation_space=XORTestRig.observation_space())
        rlagent = agent.Agent(env, actor=self._xor_actor(env), critic=self._xor_critic(env))
        warnings.simplefilter(action='ignore', category=UserWarning)
        rlagent.fit(nsteps=15000, nmaxsteps=4)
        score = rlagent.inference(nepisodes=4)
        raw_episode_reward_vals = score.history['episode_reward']
        avg_episode_reward = sum(raw_episode_reward_vals)/len(raw_episode_reward_vals)
        self.assertGreaterEqual(avg_episode_reward, 0.0)

    def test_single_step_xor_via_agent(self):
        """
        Test the single step XOR environment. This test assures us that the RL algorithm used
        actually works even if every single step is terminal.
        """
        env = environment.TestEnvironment(XORSingleStepTestRig.behavior,
                                        nsteps=1,
                                        first_obs=XORSingleStepTestRig.sample,
                                        action_shape=XORSingleStepTestRig.action_shape(),
                                        observation_space=XORSingleStepTestRig.observation_space())
        rlagent = agent.Agent(env, actor=self._xor_actor(env), critic=self._xor_critic(env))
        warnings.simplefilter(action='ignore', category=UserWarning)
        rlagent.fit(nsteps=15000, nmaxsteps=1)
        score = rlagent.inference(nepisodes=16)
        raw_episode_reward_vals = score.history['episode_reward']
        avg_episode_reward = sum(raw_episode_reward_vals)/len(raw_episode_reward_vals)
        self.assertGreaterEqual(avg_episode_reward, 2.0)

    def test_save_load_and_keep_training(self):
        """
        Test saving the XOR agent, then loading it and training it some more.
        """
        env = environment.TestEnvironment(XORTestRig.behavior,
                                          nsteps=4,
                                          first_obs=TestRL.FirstObs.XOR.value,
                                          action_shape=XORTestRig.action_shape(),
                                          observation_space=XORTestRig.observation_space())
        rlagent = agent.Agent(env, actor=self._xor_actor(env), critic=self._xor_critic(env))

        # Train for a little bit
        warnings.simplefilter(action='ignore', category=UserWarning)
        rlagent.fit(nsteps=7500, nmaxsteps=4)

        # Save the agent
        rlagent.save("tmpagent.hdf5")

        # Kill the Agent, then reload with the weights
        rlagent = None
        rlagent = agent.Agent(env, actor=self._xor_actor(env), critic=self._xor_critic(env), weights="tmpagent.hdf5")

        # Train for a little longer and make sure the reward is as good as it otherwise would have been
        os.remove("tmpagent_actor.hdf5")
        os.remove("tmpagent_critic.hdf5")
        warnings.simplefilter(action='ignore', category=UserWarning)
        rlagent.fit(nsteps=7500, nmaxsteps=4)

        # Save again
        rlagent.save("tmpagent.hdf5")

        # Kill the Agent, then reload again
        rlagent = None
        rlagent = agent.Agent(env, actor=self._xor_actor(env), critic=self._xor_critic(env), weights="tmpagent.hdf5")

        # Remove the weights files and do inference
        os.remove("tmpagent_actor.hdf5")
        os.remove("tmpagent_critic.hdf5")
        score = rlagent.inference(nepisodes=4)
        raw_episode_reward_vals = score.history['episode_reward']
        avg_episode_reward = sum(raw_episode_reward_vals)/len(raw_episode_reward_vals)
        self.assertGreaterEqual(avg_episode_reward, 2.0)

    @unittest.skipIf(__skip_the_long_ones, "")
    @unittest.skipIf("TRAVIS_CI" in os.environ, "This test takes forever. Run as part of full suite, but not part of Travis.")
    def test_somenvironment_ph0(self):
        """
        Test the SomEnvironment (the voice synthesis environment) via the Agent in phase 0.
        This attempts to produce a sound that is as loud as possible.
        """
        # Make two fake clusters
        nclusters = 2
        prototypes = [audiosegment.silent(), audiosegment.silent()]

        # We'll do this long of an articulation. Half a second is the minimum really to hear anything.
        artic_dur_ms = 1000

        # Let's do these time points for articulation control. We should really keep this number pretty small. It takes forever
        # otherwise and the learning process also takes longer, even if it has more ultimate control.
        time_point_list_ms = [0, 500, 1000]

        # Make the SOM Environment with our fake clusters. We won't use them since we'll only be using phase 0 training.
        env = environment.SomEnvironment(nclusters, artic_dur_ms, time_point_list_ms, prototypes)

        # Create the agent
        rlagent = agent.Agent(env, actor=self._som_actor(env), critic=self._som_critic(env), warmup_actor=5, warmup_critic=5)

        # Train the agent. Each step takes a long time in this environment, but we seem to be able to actually
        # teach it to make some noises pretty quickly.
        warnings.simplefilter(action='ignore', category=DeprecationWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        rlagent.fit(nsteps=400, nmaxsteps=1)

        # Do inference to get a score history. If you want to dump the audio to disk, uncomment the line before and after.
        env.retain_audio = True  # Uncomment this line and the one after inference() to dump some files for human consumption
        score = rlagent.inference(nepisodes=5)
        env.dump_audio("anysound")  # Uncomment for human consumption

        # Get the scores and compute whether we actually learned anything.
        raw_episode_reward_vals = score.history['episode_reward']
        avg_episode_reward = sum(raw_episode_reward_vals) / len(raw_episode_reward_vals)
        self.assertGreaterEqual(avg_episode_reward, 0.5)

    @unittest.skipIf(__skip_the_long_ones, "")
    @unittest.skipIf("TRAVIS_CI" in os.environ, "This test takes forever. Run as part of full suite, but not part of Travis.")
    def test_somenvironment_ph1(self):
        """
        Test the SomEnvironment (the voice synthesis environment) via the Agent, Phase 1. This
        attempts to produce a sound that is similar to one on disk.
        """
        # We only have one file that we are interested in replicating
        nclusters = 1

        # Load the file and get some info from it
        seg = audiosegment.from_file("a_simple_sound.wav")
        artic_dur_ms = seg.duration_seconds * 1000.0
        prototypes = [seg]

        # Figure out the time points that we will allow the agent to change its muscle activations
        # We will just evenly space them across time
        ntimepoints = 4
        timebase = artic_dur_ms / ntimepoints
        time_point_list_ms = [timebase * i for i in range(ntimepoints)]

        # Create the SOM Env
        env = environment.SomEnvironment(nclusters, artic_dur_ms, time_point_list_ms, prototypes)
        env.phase = 1  # Set to mimic the input file, rather than try to learn to output noise

        # Create the Agent
        rlagent = agent.Agent(env, actor=self._som_actor(env), critic=self._som_critic(env), warmup_actor=10, warmup_critic=10)

        # Train the Agent to mimic the sound as best as it can
        warnings.simplefilter(action='ignore', category=DeprecationWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        rlagent.fit(nsteps=10000, nmaxsteps=1)

        # Check on the output now that the Agent is trained
        env.retain_audio = True  # Uncomment for human consumption
        score = rlagent.inference(nepisodes=5)
        env.dump_audio("mimic")  # Uncomment for human consumption

        # Get the Rewards and assert that the Agent managed to learn something
        raw_episode_reward_vals = score.history['episode_reward']
        avg_episode_reward = sum(raw_episode_reward_vals) / len(raw_episode_reward_vals)
        self.assertGreaterEqual(avg_episode_reward, 0.5)  # TODO: this value is arbitrary right now

    def test_somenvironment_inference_mode(self):
        """
        Test to make sure we cycle through the observations on resets when in inference mode.
        """
        nclusters = 5
        assert nclusters > 0
        prototypes = [audiosegment.silent() for _ in range(nclusters)]
        artic_dur_ms = prototypes[0].duration_seconds * 1000.0

        ntimepoints = 3
        timebase = artic_dur_ms / ntimepoints
        time_point_list_ms = [timebase * i for i in range(ntimepoints)]

        env = environment.SomEnvironment(nclusters, artic_dur_ms, time_point_list_ms, prototypes)
        env.phase = 1  # Set to mimic the input file, rather than try to learn to output noise
        env.inference_mode = True

        _action = np.random.rand(*env.action_shape)

        obs = env.reset()
        self.assertTrue(np.all(obs == np.array([0])))

        obs = env.reset()
        self.assertTrue(np.all(obs == np.array([1])))

        obs = env.reset()
        self.assertTrue(np.all(obs == np.array([2])))

        obs = env.reset()
        self.assertTrue(np.all(obs == np.array([3])))

        obs = env.reset()
        self.assertTrue(np.all(obs == np.array([4])))

        obs = env.reset()
        self.assertTrue(np.all(obs == np.array([0])))

        obs = env.reset()
        self.assertTrue(np.all(obs == np.array([1])))


if __name__ == "__main__":
    unittest.main()
