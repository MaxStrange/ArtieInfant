import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
import numpy as np
import os
import rl
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.agents import DDPGAgent

class Agent:
    def __init__(self, env, actor=None, critic=None, weights=None, warmup_actor=100, warmup_critic=100, gamma=0.99):
        """
        Constructs a model to learn the given environment.

        :param env:             An environment from environment.py.
        :param actor:           The actor portion of the model or None, in which case a default is supplied.
        :param critic:          The critic portion of the model or None, in which case a default is supplied.
        :param weights:         If not None, will load the weights from two files named <weights>_actor<extension>
                                and <critic>_critic<extension>.
        :param warmup_actor:    The number of steps to take before training the actor.
        :param warmup_critic:   The number of steps to take before training the critic.
        :param gamma:           The discount factor.
        """
        self.env             = env
        nactions             = np.product(self.env.action_shape)
        actor                = self._actor() if actor is None else actor
        critic, action_input = self._critic() if critic is None else critic
        membuf               = SequentialMemory(int(1E5), window_length=1)
        random               = OrnsteinUhlenbeckProcess(size=nactions, theta=0.15, mu=0.0, sigma=0.3)
        self.agent           = DDPGAgent(nb_actions=nactions,
                                         actor=actor,
                                         critic=critic,
                                         critic_action_input=action_input,
                                         memory=membuf,
                                         nb_steps_warmup_critic=warmup_critic,
                                         nb_steps_warmup_actor=warmup_actor,
                                         random_process=random,
                                         gamma=gamma,
                                         target_model_update=1E-3)
        self.agent.compile(keras.optimizers.Adam(lr=0.001, clipnorm=1.0), metrics=['mae', 'accuracy'])

        # Potentially load the Agent's weights from disk
        if weights is not None:
            basename, ext = os.path.splitext(weights)
            cweights = basename + "_critic" + ext
            aweights = basename + "_actor" + ext
            if not os.path.isfile(cweights):
                raise ValueError("Could not find file", cweights)
            elif not os.path.isfile(aweights):
                raise ValueError("Could not find file", aweights)
            else:
                self.agent.load_weights(weights)

    def fit(self, nsteps=50000, nmaxsteps=200):
        """
        Train the agent on the environment.

        :param nsteps:      The number of steps in total to take over the whole training regimen.
        :param nmaxsteps:   The maximum number of steps to take in a single episode before resetting the environment.
        :returns:           A keras.callbacks.history object.
        """
        return self.agent.fit(self.env, nb_steps=nsteps, visualize=False, verbose=1, nb_max_episode_steps=nmaxsteps)

    def save(self, fpath):
        """
        Saves the Agent into a file at `fpath`.
        Uses hdf5.

        :param fpath:       The path to save the file.
        """
        self.agent.save_weights(fpath, overwrite=True)

    def inference(self, nepisodes=50, nmaxsteps=200):
        """
        Does inference on the environment.

        :param nepisodes:       The number of episodes to do inference on.
        :param nmaxsteps:       The maximum number of steps per episode before resetting environment.
        :returns:               A keras.callbacks.history object.
        """
        return self.agent.test(self.env, nb_episodes=nepisodes, visualize=False, nb_max_episode_steps=nmaxsteps)

    def _actor(self):
        """
        Construct the actor part of the model and return it.
        """
        nactions = np.product(self.env.action_shape)

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=(1,) + self.env.observation_space.shape))
        model.add(keras.layers.Dense(16))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(8))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(nactions))
        model.add(keras.layers.Activation('linear'))
        return model

    def _critic(self):
        """
        Construct the critic part of the model and return it.
        """
        nactions      = np.product(self.env.action_shape)
        action_input  = keras.layers.Input(shape=(nactions,), name='action_input')
        obs_input     = keras.layers.Input(shape=(1,) + self.env.observation_space.shape, name='observation_input')
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
