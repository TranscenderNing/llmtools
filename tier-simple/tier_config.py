import json
import os


class TierConfig:
    def __init__(
        self,
        op_position="ffn",
        intervention_type="red",
        intervention_params={"hidden_size": 4096},
    ):
        self.op_position = op_position
        self.intervention_type = intervention_type
        self.intervention_params = intervention_params

    @staticmethod
    def from_pretrained(load_directory):
        saved_config = json.load(open(os.path.join(load_directory, "config.json"), "r"))
        tier_config = TierConfig(
            op_position=saved_config["op_position"],
            intervention_type=saved_config["intervention_type"],
            intervention_params=saved_config["intervention_params"],
        )
        return tier_config

    def save_pretrained(self, save_directory):
        config_dict = {}
        config_dict["op_position"] = self.op_position
        config_dict["intervention_type"] = self.intervention_type
        config_dict["intervention_params"] = self.intervention_params
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
