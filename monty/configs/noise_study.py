# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Experiment configs for exploratory noise study."""

from copy import deepcopy

from .common import make_randrot_variant
from .fig3_robust_sensorimotor_inference import dist_agent_1lm

# Gets added to by `make_and_add_config`.
CONFIGS = {}


def make_and_add_config(noise_params: dict, run_name: str):
    """Create a 1-LM distant agent agent experiment with given noise parameters.

    Creates experiment configs with:
     - Distant agent
     - 1 LM
     - 5 predefined random rotations
     - varying levels of sensor noise

    The config is added to `CONFIGS` with the given `run_name`.
    """
    # Make a copy of the distant agent config with 5 random rotations.
    config = make_randrot_variant(dist_agent_1lm, run_name)

    # Add sensor noise.
    for sm_dict in config["monty_config"].sensor_module_configs.values():
        sm_args = sm_dict["sensor_module_args"]
        if sm_args["sensor_module_id"] != "view_finder":
            sm_args["noise_params"] = deepcopy(noise_params)

    # Add the config to `CONFIGS`.
    CONFIGS[run_name] = config


all_noise_params = {
    "none": {},
    "loc": {"location": 0.002},
    "hsv": {"features": {"hsv": 0.1}},
    "hsv5": {"features": {"hsv": 0.5}},
    "loc_hsv": {"location": 0.002, "features": {"hsv": 0.1}},
    "loc_hsv5": {"location": 0.002, "features": {"hsv": 0.5}},
    "std": {
        "location": 0.002,
        "features": {
            "pose_vectors": 2.0,
            "hsv": 0.1,
            "principal_curvatures_log": 0.1,
            "pose_fully_defined": 0.01,
        },
    },
    "std_hsv5": {
        "location": 0.002,
        "features": {
            "pose_vectors": 2.0,
            "hsv": 0.5,
            "principal_curvatures_log": 0.1,
            "pose_fully_defined": 0.01,
        },
    },
}

for name, params in all_noise_params.items():
    make_and_add_config(params, f"dist_agent_1lm_{name}")
