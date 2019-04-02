import gym
from gym.envs.registration import register

from .cliff_walking import *
from .frozen_lake import *

__all__ = ['RewardingFrozenLakeEnv', 'WindyCliffWalkingEnv']

register(
    id='RewardingFrozenLake-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='RewardingFrozenLake8x8less03-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'step_reward': -0.3}
)

register(
    id='RewardingFrozenLake8x8less05-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'step_reward': -0.5}
)

register(
    id='RewardingFrozenLake8x8less09-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'step_reward': -0.9}
)

register(
    id='RewardingFrozenLake8x8more01-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'step_reward': 0.1}
)

register(
    id='RewardingFrozenLake8x8more05-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'step_reward': 0.5}
)

register(
    id='RewardingFrozenLake8x8more09-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'step_reward': 0.9}
)

register(
    id='RewardingFrozenLake8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8'}
)

register(
    id='RewardingFrozenLakeNoRewards20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': False}
)

register(
    id='RewardingFrozenLakeNoRewards8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'rewarding': False}
)

register(
    id='RewardingFrozenLake20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': False}
)

register(
    id='WindyCliffWalking-v0',
    entry_point='environments:WindyCliffWalkingEnv',
)

register(
    id='WindyCliffWalking_03-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'wind_prob': 0.3}
)

register(
    id='WindyCliffWalking_05-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'wind_prob': 0.5}
)

register(
    id='WindyCliffWalking_07-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'wind_prob': 0.7}
)

register(
    id='WindyCliffWalking_09-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'wind_prob': 0.9}
)

register(
    id='RewardingFrozenLake20x20less01-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'step_reward': -0.1}
)

register(
    id='RewardingFrozenLake20x20less05-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'step_reward': -0.5}
)

register(
    id='RewardingFrozenLake20x20less09-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'step_reward': -0.9}
)

register(
    id='RewardingFrozenLake20x20more01-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'step_reward': 0.1}
)

register(
    id='RewardingFrozenLake20x20more05-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'step_reward': 0.5}
)

register(
    id='RewardingFrozenLake20x20more09-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'step_reward': 0.9}
)

def get_rewarding_frozen_lake_environment():
    return gym.make('RewardingFrozenLake8x8-v0')


def get_rewarding_frozen_lake_environment_less_03_v0():
    return gym.make('RewardingFrozenLake8x8less03-v0')

def get_rewarding_frozen_lake_environment_less_05_v0():
    return gym.make('RewardingFrozenLake8x8less05-v0')

def get_rewarding_frozen_lake_environment_less_09_v0():
    return gym.make('RewardingFrozenLake8x8less09-v0')

def get_rewarding_frozen_lake_environment_more_01_v0():
    return gym.make('RewardingFrozenLake8x8more01-v0')

def get_rewarding_frozen_lake_environment_more_05_v0():
    return gym.make('RewardingFrozenLake8x8more05-v0')

def get_rewarding_frozen_lake_environment_more_09_v0():
    return gym.make('RewardingFrozenLake8x8more09-v0')

def get_rewarding_frozen_lake_environment_less_01_20by20_v0():
    return gym.make('RewardingFrozenLake20x20less01-v0')

def get_rewarding_frozen_lake_environment_less_05_20by20_v0():
    return gym.make('RewardingFrozenLake20x20less05-v0')

def get_rewarding_frozen_lake_environment_less_09_20by20_v0():
    return gym.make('RewardingFrozenLake20x20less09-v0')

def get_rewarding_frozen_lake_environment_more_01_20by20_v0():
    return gym.make('RewardingFrozenLake20x20more01-v0')

def get_rewarding_frozen_lake_environment_more_05_20by20_v0():
    return gym.make('RewardingFrozenLake20x20more05-v0')

def get_rewarding_frozen_lake_environment_more_09_20by20_v0():
    return gym.make('RewardingFrozenLake20x20more09-v0')

def get_large_rewarding_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards20x20-v0')


def get_frozen_lake_environment():
    return gym.make('FrozenLake-v0')


def get_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards8x8-v0')


def get_large_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards20x20-v0')


def get_cliff_walking_environment():
    return gym.make('CliffWalking-v0')

def get_cliff_walking_environment_03():
    return gym.make('WindyCliffWalking_03-v0')

def get_cliff_walking_environment_05():
    return gym.make('WindyCliffWalking_05-v0')

def get_cliff_walking_environment_07():
    return gym.make('WindyCliffWalking_07-v0')

def get_cliff_walking_environment_09():
    return gym.make('WindyCliffWalking_09-v0')

def get_windy_cliff_walking_environment():
    return gym.make('WindyCliffWalking-v0')
