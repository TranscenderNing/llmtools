import torch

config_dict = {
    "representations": [
        {"layer": 0, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 1, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 2, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 3, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 4, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 5, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 6, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 7, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 8, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 9, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 10, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 11, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 12, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 13, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 14, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 15, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 16, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 17, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 18, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 19, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 20, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 21, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 22, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 23, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 24, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 25, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 26, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 27, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 28, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 29, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 30, "component": "block_output", "low_rank_dimension": 8},
        {"layer": 31, "component": "block_output", "low_rank_dimension": 8},
    ],
    "intervention_params": {
        "embed_dim": 4096,
        "low_rank_dimension": 8,
        "dropout": 0.0,
        "dtype": torch.bfloat16,
        "act_fn": None,
        "device": "cuda",
        "add_bias": False,
    },
    "intervention_types": [
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
        "<class 'tier_interventions.LoreftIntervention'>",
    ],
    "position": "f7+l7",
}

for key, val in config_dict["intervention_params"].items():
    if not isinstance(val, str):
        config_dict["intervention_params"][key] = repr(val)


import os
import json

save_directory = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex"
with open(os.path.join(save_directory, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=4)
