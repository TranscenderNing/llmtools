import copy
import logging
import os
import re

import torch
import torch.nn as nn
from interventions import get_intervene_module, intervention_mapping
from tier_config import TierConfig

target_dict = {}
total_parameter = 0


class RepresentationLayer(nn.Module):
    def __init__(
        self,
        original_layer,
        op_position="ffn",
        intervention_type="red",
        intervention_params={},
    ):
        super().__init__()
        self.original_layer = original_layer
        self.op_position = op_position
        # new module
        self.intervene_module = get_intervene_module(
            intervention_type, intervention_params
        )
        print("op positon", self.op_position)
        print("original layer", self.original_layer)

    def forward(self, x, input_tensor=None):
        if (
            self.op_position == "res"
            or self.op_position == "res_with_attn"
            or self.op_position == "res_with_res"
        ):
            hidden_states = self.original_layer(x, input_tensor)
        else:
            hidden_states = self.original_layer(x)

        hidden_states = self.intervene_module(hidden_states)
        return hidden_states


class RepresentationModel(nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, base_model, tier_config):
        super().__init__()
        self.tier_config = tier_config
        self.base_model = base_model
        self.op_position = tier_config.op_position
        self.intervention_type = tier_config.intervention_type
        self.intervention_params = tier_config.intervention_params
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if self.check_update(key):
                self.replace_layer(key)

    def check_update(self, key):
        if self.op_position == "ffn":
            return self.match_substring(key)
        elif self.op_position == "post_attention_layernorm":
            return self.match_post_norm_substring(key)

    def generate(self, **args):
        return self.base_model.generate(**args)

    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = RepresentationLayer(
            original_layer=replaced_module,
            op_position=self.op_position,
            intervention_type=self.intervention_type,
            intervention_params=self.intervention_params,
        )
        setattr(parent_module, replaced_name_last, new_module)

    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0
        for name, param in self.base_model.named_parameters():
            total_parameters += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()

        return {
            "total_para:": total_parameters,
            "trainable_para: ": trainable_parameters,
            "trainable%:": f"{100 * trainable_parameters / total_parameters:.4f}",
        }

    def frozen_model(self):
        for _, param in self.base_model.named_parameters():
            param.requires_grad = False

    def match_substring(self, input_string):
        pattern = r"down_proj"
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False

    def match_post_norm_substring(self, input_string):
        pattern = "post_attention_layernorm"
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output

    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                if "activation_ln" in key:
                    if "weight" in replaced_name_last:
                        self.base_model.get_submodule(
                            parent_key
                        ).weight.data = new_module
                    elif "bias" in replaced_name_last:
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[
                        replaced_name_last
                    ] = new_module

    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, os.path.join(save_path, "delta_vector.pth"))

    def save_pretrained(self, save_directory, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            logging.info(f"Directory '{save_directory}' created successfully.")
        else:
            logging.info(f"Directory '{save_directory}' already exists.")
        saving_config = copy.deepcopy(self.tier_config)
        saving_config.save_pretrained(save_directory)

        intervention_class = intervention_mapping[self.intervention_type]
        for name, module in self.base_model.named_modules():
            if isinstance(module, intervention_class):
                binary_filename = f"{name}.bin"
                # save intervention binary file
                print(f"Saving trainable intervention to {binary_filename}.")
                torch.save(
                    module.state_dict(),
                    os.path.join(save_directory, binary_filename),
                )

    def from_pretrained(load_directory, model):
        tier_config = TierConfig.from_pretrained(
            load_directory=load_directory,
        )
        model = RepresentationModel(model, tier_config)
        intervention_class = tier_config.intervention_type
        # load binary files
        for name, module in model.named_modules():
            if isinstance(module, intervention_class):
                binary_filename = f"{name}.bin"
                binary_filename = f"{name}.bin"
                saved_state_dict = torch.load(
                    os.path.join(load_directory, binary_filename)
                )
                module.load_state_dict(saved_state_dict)
