from __future__ import absolute_import, division, print_function, unicode_literals

from .grid_world_environment import create_grid_world_env
from .wrappers import DictObservationNormalizationWrapper, GridWorldEvaluatorWrapper


def make_environment(env_cfg, lock=None):
    """ Use this to make Environments """

    env_name = env_cfg.env_name

    assert env_name.startswith("MarlGrid")
    env = create_grid_world_env(env_cfg)
    env = GridWorldEvaluatorWrapper(env)
    env = DictObservationNormalizationWrapper(env)

    return env
