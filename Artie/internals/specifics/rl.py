"""
These are the functions for the specific usage of the RL package.
"""
from internals.rl import agent        # pylint: disable=locally-disabled, import-error
from internals.rl import environment  # pylint: disable=locally-disabled, import-error
import audiosegment
import keras
import numpy as np
import warnings

def _actor(env):
    """
    Construct the actor portion of the DDPG model and return it.
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

def _critic(env):
    """
    Construct the critic portion of the DDPG model and return it.
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

def pretrain(config):
    """
    Create an instance of the SomEnvironment and train an agent
    to make noise. Save the weights for the actor and critic
    networks.

    :param config:  The experiment configuration file.
    """
    # We don't cluster while pretraining, but we need a fake cluster list to make the env happy
    nclusters = 1
    prototypes = [audiosegment.silent()]

    # Determine how long all articulations will be
    artic_dur_ms = config.getint('synthesizer', 'phoneme-durations-ms')

    # Determine when in the articulation the muscle activations will be controllable
    time_point_list_ms = config.getlist('synthesizer', 'articulation-time-points-ms', type=int)

    # Make the SOM Environment with our fake clusters. We won't use them since we'll only be using phase 0 training.
    env = environment.SomEnvironment(nclusters, artic_dur_ms, time_point_list_ms, prototypes)

    # Create the agent
    warmup_actor  = config.getint('synthesizer', 'actor-warmup-steps')
    warmup_critic = config.getint('synthesizer', 'critic-warmup-steps')
    gamma         = config.getfloat('synthesizer', 'discount-factor')
    actor         = _actor(env)
    critic        = _critic(env)
    rlagent = agent.Agent(env, actor=actor, critic=critic, warmup_actor=warmup_actor, warmup_critic=warmup_critic, gamma=gamma)

    # Train the agent. Each step takes a long time in this environment, but we seem to be able to actually
    # teach it to make some noises pretty quickly.
    nsteps = config.getint('synthesizer', 'pretraining-steps')
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    rlagent.fit(nsteps=nsteps, nmaxsteps=1)

    # Save the weights and return the right stuff
    basename = "pretrained_synthesis_weights"
    rlagent.save(basename + ".hdf5")
    return basename, actor, critic
