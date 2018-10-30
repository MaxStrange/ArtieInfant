"""
Test the reinforcement learning algorithm.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras

import argparse
import audiosegment
import enum
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import internals.rl.agent as agent              # pylint: disable=locally-disabled, import-error
import internals.rl.environment as environment  # pylint: disable=locally-disabled, import-error


def _som_actor(env):
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

def _som_critic(env):
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

def _make_environment():
    """
    Make and return a default SOM Environment suitable for the testing. It uses two
    files to try to mimic.
    """
    # We have two files that we want to replicate
    nclusters = 2

    # Load the files and figure out the length of the longest one. That's how long all our
    # articulations will be
    seg0 = audiosegment.from_file("a_simple_sound.wav")
    seg1 = audiosegment.from_file("another_simple_sound.wav")
    prototypes = [seg0, seg1]
    artic_dur_ms = max([seg.duration_seconds * 1000.0 for seg in prototypes])

    # Figure out the time points that we will allow the agent to change its muscle activations
    # We will just evenly space them across time
    ntimepoints = 3
    timebase = artic_dur_ms / ntimepoints
    time_point_list_ms = [timebase * i for i in range(ntimepoints)]

    # Create the SOM Env
    env = environment.SomEnvironment(nclusters, artic_dur_ms, time_point_list_ms, prototypes)
    env.phase = 1  # Set to mimic the input file, rather than try to learn to output noise

    return env

def train():
    """
    Test the SomEnvironment (the voice synthesis environment) via the Agent, Phase 1. This
    attempts to produce a sound that is similar to one on disk.
    """
    env = _make_environment()

    # Create the Agent
    rlagent = agent.Agent(env, actor=_som_actor(env), critic=_som_critic(env), warmup_actor=10, warmup_critic=10)

    # Train the Agent to mimic the sound as best as it can
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    rlagent.fit(nsteps=10_000, nmaxsteps=1)

    # Save the agent's weights
    rlagent.save("weights")

    # Check on the output now that the Agent is trained
    env.retain_audio = True  # Uncomment for human consumption
    score = rlagent.inference(nepisodes=10)
    env.dump_audio("mimic")  # Uncomment for human consumption

    # Get the Rewards
    raw_episode_reward_vals = score.history['episode_reward']
    avg_episode_reward = sum(raw_episode_reward_vals) / len(raw_episode_reward_vals)
    print("AVG REWARD:", avg_episode_reward)

def inference():
    """
    Do inference with a trained agent by loading the agent's weights into a new agent,
    and then setting up an inference environment that just cycles through the observations
    rather than drawing from them uniformly.
    """
    env = _make_environment()
    env.inference_mode = True

    # Create the Agent
    rlagent = agent.Agent(env, actor=_som_actor(env), critic=_som_critic(env), warmup_actor=10, warmup_critic=10, weights="weights")

    # Do inference for 2 steps on the environment. This should give you both of the agent's actions.
    env.retain_audio = True
    score = rlagent.inference(nepisodes=2)
    env.dump_audio("mimicry_inference")

    # Get the Rewards
    raw_episode_reward_vals = score.history['episode_reward']
    avg_episode_reward = sum(raw_episode_reward_vals) / len(raw_episode_reward_vals)
    print("AVG REWARD:", avg_episode_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test instead of train")
    args = parser.parse_args()

    if args.test:
        inference()
    else:
        train()

