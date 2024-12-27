import copy
import json
import logging
import os
import types
from typing import List, Optional

from utils import (
    HandlerList,
    count_parameters,
    create_directory,
    do_intervention,
    gather_neurons,
    get_module_hook,
    scatter_neurons,
)
from tier_config import TierConfig
from torch import nn
import torch


class TierModel(nn.Module):
    """
    config: TierConfig
    """

    def __init__(self, config, model, **kwargs):
        super().__init__()
        self.config = config
        self.num_interventions = len(config.representations)
        self.position = config.position
        self.intervention_types = config.intervention_types
        self.representations = {}
        self.interventions = {}
        _original_key_order = []
        # for generate
        self._key_setter_call_counter = {}
        for i, representation in enumerate(config.representations):
            _key = f'layer.{representation["layer"]}'
            if representation["intervention"] is not None:
                intervention = representation["intervention"]

            module_hook = get_module_hook(model, representation)
            self.representations[_key] = representation
            self.interventions[_key] = (intervention, module_hook)
            _original_key_order += [_key]

            # usually, it's a one time call per
            # hook unless model generates.
            self._key_setter_call_counter[_key] = 0

        self.sorted_keys = _original_key_order
        self.model = model
        self.model_config = model.config
        self.disable_model_gradients()
        self.trainable_model_parameters = {}

    def forward(
        self,
        **base,
    ):
        unit_locations = base["intervention_locations"].permute(1, 0, 2).tolist()
        self._reset_hook_count()
        try:
            # intervene, register hook after decoder block
            set_handlers_to_remove = self._wait_for_forward_with_intervention(
                unit_locations
            )
            # run intervened forward
            del base["intervention_locations"]
            del base["id"]
            counterfactual_outputs = self.model(**base)
            set_handlers_to_remove.remove()
        except Exception as e:
            raise e
        self._reset_hook_count()
        return counterfactual_outputs

    def generate(
        self,
        base,
        unit_locations: Optional[List] = None,
        intervene_on_prompt: bool = False,
        output_original_output: Optional[bool] = False,
        **kwargs,
    ):
        self._reset_hook_count()
        self._intervene_on_prompt = intervene_on_prompt
        base_outputs = None
        if output_original_output or True:
            # returning un-intervened output
            base_outputs = self.model.generate(**base, **kwargs)
        set_handlers_to_remove = None
        try:
            # intervene, register hook after decoder block
            set_handlers_to_remove = self._wait_for_forward_with_intervention(
                unit_locations,
            )
            # run intervened generate
            counterfactual_outputs = self.model.generate(**base, **kwargs)
            set_handlers_to_remove.remove()
        except Exception as e:
            raise e
        self._reset_hook_count()
        return base_outputs, counterfactual_outputs

    def _wait_for_forward_with_intervention(
        self,
        unit_locations,
    ):
        all_set_handlers = HandlerList([])
        for key_id, key in enumerate(self.sorted_keys):
            set_handlers = self._intervention_setter(key, unit_locations[key_id])
            all_set_handlers.extend(set_handlers)
        return all_set_handlers

    def _intervention_setter(
        self,
        key,
        unit_locations_base,
    ) -> HandlerList:
        """
        Create a list of setter handlers that will set activations
        """
        handlers = []
        intervention, module_hook = self.interventions[key]

        def hook_callback(
            model,
            inputs,
            outputs,
        ):
            is_prompt = self._key_setter_call_counter[key] == 0
            if is_prompt:
                self._key_setter_call_counter[key] += 1
            if not is_prompt:
                return

            selected_output = self._gather_intervention_output(
                outputs, key, unit_locations_base
            )

            if not isinstance(self.interventions[key][0], types.FunctionType):
                intervened_representation = do_intervention(
                    selected_output,
                    intervention,
                )
            if intervened_representation is None:
                return

            if isinstance(outputs, tuple):
                _ = self._scatter_intervention_output(
                    outputs[0],
                    intervened_representation,
                    key,
                    unit_locations_base,
                )
            else:
                _ = self._scatter_intervention_output(
                    outputs,
                    intervened_representation,
                    key,
                    unit_locations_base,
                )

        handlers.append(
            module_hook(
                hook_callback,
            )
        )

        return HandlerList(handlers)

    def _gather_intervention_output(
        self, output, representations_key, unit_locations
    ) -> torch.Tensor:
        """
        Gather intervening activations from the output based on indices
        """
        if isinstance(output, tuple):
            original_output = output[0].clone()
        else:
            original_output = output.clone()
        if unit_locations is None:
            return original_output

        # gather based on intervention locations
        selected_output = gather_neurons(
            original_output,
            unit_locations,
        )
        return selected_output

    def _scatter_intervention_output(
        self,
        output,
        intervened_representation,
        representations_key,
        unit_locations,
    ) -> torch.Tensor:
        """
        Scatter in the intervened activations in the output
        """
        # data structure casting
        if isinstance(output, tuple):
            original_output = output[0]
        else:
            original_output = output
        # for non-sequence-based models, we simply replace
        # all the activations.
        if unit_locations is None:
            original_output[:] = intervened_representation[:]
            return original_output

        # component = self.representations[representations_key].component
        # unit = self.representations[representations_key].unit

        # scatter in-place
        _ = scatter_neurons(
            original_output,
            intervened_representation,
            unit_locations,
        )

        return original_output

    def save_pretrained(self, save_directory, **kwargs):
        create_directory(save_directory)
        saving_config = copy.deepcopy(self.config)
        saving_config.sorted_keys = self.sorted_keys
        saving_config.intervention_types = []
        saving_config.intervention_dimensions = []

        for k, v in self.interventions.items():
            intervention = v[0]
            saving_config.intervention_types += [(type(intervention))]
            binary_filename = f"intkey_{k}.bin"
            # save intervention binary file
            print(f"Saving trainable intervention to {binary_filename}.")
            torch.save(
                intervention.state_dict(),
                os.path.join(save_directory, binary_filename),
            )

        saving_config.save_pretrained(save_directory)

    @staticmethod
    def from_pretrained(
        load_directory,
        model,
    ):
        reft_config = TierConfig.from_pretrained(
            load_directory=load_directory,
        )
        intervenable = TierModel(reft_config, model)
        intervenable.disable_model_gradients()

        # load binary files
        for i, (k, v) in enumerate(intervenable.interventions.items()):
            intervention = v[0]
            binary_filename = f"intkey_{k}.bin"
            saved_state_dict = torch.load(os.path.join(load_directory, binary_filename))
            intervention.load_state_dict(saved_state_dict)
        return intervenable

    # def train(self):
    #     self.model.train()
        
    def train(self, mode=True):
        self.model.train(mode=mode)

    def eval(self):
        self.model.eval()

    def count_parameters(self, include_model=False):
        total_parameters = 0
        for k, v in self.interventions.items():
            total_parameters += count_parameters(v[0])
        if include_model:
            total_parameters += sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        return total_parameters

    def print_trainable_parameters(self):
        trainable_intervention_parameters = 0
        for k, v in self.interventions.items():
            trainable_intervention_parameters += count_parameters(v[0])

        trainable_model_parameters = int(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )

        all_model_parameters = int(sum(p.numel() for p in self.model.parameters()))

        total_trainable_parameters = (
            trainable_intervention_parameters + trainable_model_parameters
        )

        print("trainable_intervention_parameters:", trainable_intervention_parameters)
        print("trainable_model_parameters:", trainable_model_parameters)
        print("all_model_parameters:", all_model_parameters)
        print("total_trainable_parameters:", total_trainable_parameters)
        print(
            f"trainable intervention params: {trainable_intervention_parameters:,d} || trainable model params: {trainable_model_parameters:,d}\n"
            f"model params: {all_model_parameters:,d} || trainable%: {100 * total_trainable_parameters / all_model_parameters}"
        )

    def _reset_hook_count(self):
        """
        Reset the hook count before any generate call
        """
        self._key_setter_call_counter = dict.fromkeys(self._key_setter_call_counter, 0)

    def __str__(self):
        attr_dict = {
            "model_type": str(self.model_type),
            "intervention_types": str(self.intervention_types),
            "alignabls": self.sorted_keys,
        }
        return json.dumps(attr_dict, indent=4)

    def get_trainable_parameters(self):
        """
        Return trainable params as key value pairs
        """
        ret_params = []
        for k, v in self.interventions.items():
            ret_params += [p for p in v[0].parameters()]
        for p in self.model.parameters():
            if p.requires_grad:
                ret_params += [p]
        return ret_params

    def named_parameters(self, recurse=True, include_sublayers=True):
        """
        The above, but for HuggingFace.
        """
        ret_params = []
        for k, v in self.interventions.items():
            ret_params += [(k + "." + n, p) for n, p in v[0].named_parameters()]
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                ret_params += [("model." + n, p)]
        return ret_params

    def enable_model_gradients(self):
        """
        Enable gradient in the model
        """
        # Unfreeze all model weights
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        self.model_has_grad = True

    def disable_model_gradients(self):
        """
        Disable gradient in the model
        """
        # Freeze all model weights
        print("freeze model weights")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model_has_grad = False

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name + " requires grad")
