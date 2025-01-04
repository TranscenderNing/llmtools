import os
import torch
import torch.nn as nn
import re
from token_former import get_test_pattention

target_dict = {}
total_parameter = 0

class RepresentationLayer(nn.Module):
    def __init__(self, hidden_size, original_layer, layer_type="all", op_position="ffn"):
        super().__init__()
        self.original_layer = original_layer
        self.layer_type = layer_type
        self.op_position = op_position
        self.weight_type = torch.bfloat16
        # new module
        self.patten = get_test_pattention()
        self.weight = torch.rand(1)
        print('op positon', self.op_position)
        print('original layer', self.original_layer)

    def forward(self, x, input_tensor=None):
        if(self.op_position == "res" or self.op_position =="res_with_attn" or self.op_position =="res_with_res"):
            hidden_states = self.original_layer(x, input_tensor)
        else:
            hidden_states = self.original_layer(x)

        hidden_states = self.patten(hidden_states) 

        return hidden_states
    
    
    
class RepresentationLayer_RED(nn.Module):
    def __init__(self, hidden_size, original_layer, layer_type="all", op_position="ffn"):
        super().__init__()
        self.original_layer = original_layer
        self.layer_type = layer_type
        self.op_position = op_position
        self.weight_type = torch.bfloat16
        # new module
        self.delta_vector = nn.ParameterDict({
            "activation_scaling": nn.Parameter(torch.ones(1, hidden_size)),
            "activation_bias":nn.Parameter(torch.zeros(1, hidden_size)),
        })
        self.weight = torch.rand(1)
        self.delta_vector.to(self.weight_type)

    def forward(self, x, input_tensor=None):
        if(self.op_position == "res" or self.op_position =="res_with_attn" or self.op_position =="res_with_res"):
            hidden_states = self.original_layer(x, input_tensor)
        else:
            hidden_states = self.original_layer(x)

        hidden_states = hidden_states * self.delta_vector["activation_scaling"]
        hidden_states = hidden_states + self.delta_vector["activation_bias"]

        return hidden_states

class RepresentationLLama(nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]
    def __init__(self, base_model, op_position="ffn", layer_type="all", exclude_layers=[]):
        super().__init__()
        self.base_model = base_model
        self.model_type = "llama-7b"
        self.layer_type = layer_type
        self.op_position = op_position
        self.exclude_layers = exclude_layers
        if(exclude_layers):
            pattern_str = '|'.join(map(str, exclude_layers))
            pattern = re.compile(r'\b(?:' + pattern_str + r')\b')
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if(exclude_layers):
                match = pattern.search(key)
                if(match):
                    continue
            if(self.check_update(key)):
                self.replace_layer(key)   

        print(self.print_trainable_parameters())



    def check_update(self, key):
        if(self.op_position=="ffn"):
            return self.match_substring(key)
        elif(self.op_position=="post_attention_layernorm"):
            return self.match_post_norm_substring(key)

    def generate(self, **args):
        return self.base_model.generate(**args)


    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = RepresentationLayer(
                        hidden_size = self.base_model.config.hidden_size,
                        original_layer = replaced_module,
                        layer_type = self.layer_type,
                        op_position = self.op_position,
                        )
        setattr(parent_module, replaced_name_last, new_module)

    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0 
        for name, param in self.base_model.named_parameters():
            total_parameters+=param.numel()
            if(param.requires_grad):
                trainable_parameters+=param.numel()
    
        return {
            "total_para:": total_parameters,
            "trainable_para: ":trainable_parameters,
            "trainable%:" : f"{100 * trainable_parameters / total_parameters:.4f}"
            }


    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
    

    def match_substring(self, input_string):
        pattern = r'down_proj'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
    
    def match_post_norm_substring(self, input_string):
        pattern = 'post_attention_layernorm'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
        

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return  output
    
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
                if("activation_ln" in key):
                    if("weight" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).weight.data = new_module
                    elif("bias" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_peft_model(self, save_path):
        self.base_model.save_pretrained(save_path)

    def save_model(self, save_path):
        self.save_peft_model(save_path)
        save_dict = self.get_save_dict()
        torch.save(save_dict, os.path.join(save_path, "delta_vector.pth"))


