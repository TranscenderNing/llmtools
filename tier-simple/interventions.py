import torch
from torch import nn


class RedIntervention(nn.Module):
    def __init__(self, **kwargs):
        super(RedIntervention, self).__init__()
        hidden_size = kwargs["hidden_size"]
        self.delta_vector = nn.ParameterDict(
            {
                "activation_scaling": nn.Parameter(
                    torch.ones(size=(1, hidden_size), dtype=kwargs["dtype"])
                ),
                "activation_bias": nn.Parameter(
                    torch.zeros(size=(1, hidden_size), dtype=kwargs["dtype"])
                ),
            }
        )
        self.delta_vector["activation_scaling"]
        self.delta_vector["activation_bias"]

    def forward(self, x):
        x = (
            x * self.delta_vector["activation_scaling"]
            + self.delta_vector["activation_bias"]
        )
        return x


intervention_mapping = {
    "red": RedIntervention,
}


def get_intervene_module(intervention_type, intervention_params):
    return intervention_mapping[intervention_type](**intervention_params)
