import torch
from torch import nn
import torch.nn.functional as F
import math
from inspect import isfunction
from collections import OrderedDict
from transformers.activations import ACT2FN


class LowRankRotateLayer(nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super(LoreftIntervention, self).__init__()
        self.embed_dim = kwargs["embed_dim"]
        rotate_layer = LowRankRotateLayer(
            kwargs["embed_dim"], kwargs["low_rank_dimension"], init_orth=True
        ).to(kwargs["device"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer).to(
            kwargs["device"]
        )
        self.learned_source = (
            torch.nn.Linear(kwargs["embed_dim"], kwargs["low_rank_dimension"])
            .to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
            .to(kwargs["device"])
        )
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        ).to(kwargs["device"])
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base),
            self.rotate_layer.weight.T,
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Overwrite for data-efficiency.
        """
        # state_dict = state_dict.to("cuda") if isinstance(state_dict, torch.Tensor) else {k: v.to("cuda") for k, v in state_dict.items()}
        self.learned_source.load_state_dict(state_dict, strict=False)
        self.learned_source.to("cuda")

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True
        ).to(self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[
            :, :overload_w_width
        ] = overload_w
        assert (
            torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True
        )  # we must match!

        return


class LoreftIntervention_v2(nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh) + W^T(Rh - wh -b)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base):
        # print('base shape', base.shape)
        rotated_base = self.rotate_layer(base)
        output = (
            base
            + torch.matmul(
                (self.act_fn(self.learned_source(base)) - rotated_base),
                self.rotate_layer.weight.T,
            )
            + torch.matmul(
                (rotated_base - self.act_fn(self.learned_source(base))).to(base.dtype),
                self.learned_source.weight,
            )
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True
        ).to(self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[
            :, :overload_w_width
        ] = overload_w
        assert (
            torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True
        )  # we must match!

        return


class MloraIntervention(nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh) + W^T(Rh - wh -b)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)

        self.r = kwargs["low_rank_dimension"]
        self.learned_source = torch.nn.Linear(
            kwargs["low_rank_dimension"], kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, x):
        in_f = out_f = self.embed_dim
        r = self.r
        sum_inter = in_f // r
        rb1 = in_f // r if in_f % r == 0 else in_f // r + 1
        if in_f % r != 0:
            pad_size = r - in_f % r
            x = torch.cat([x, x[..., :pad_size]], dim=-1)
            sum_inter += 1
        in_x = x.view(*x.shape[:-1], sum_inter, r)
        if not hasattr(self, "cos") and not hasattr(self, "sin"):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, r, 2).float() / r))
            t = torch.arange(rb1)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
            self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
        rh_in_x = torch.cat((-in_x[..., r // 2 :], in_x[..., : r // 2]), dim=-1)
        in_x = in_x * self.cos + rh_in_x * self.sin

        out_x = self.learned_source(in_x)
        out_x = out_x.view(*x.shape[:-1], -1)[..., :out_f]

        output = x + out_x
        return self.dropout(output.to(x.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.

        return


class MosLoRAIntervention(nn.Module):
    """
    LoReFT(h) = h + R^T W' (Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)

        self.mix = torch.nn.Linear(
            kwargs["low_rank_dimension"], kwargs["low_rank_dimension"], bias=False
        )
        # Kaiming 初始化
        nn.init.kaiming_uniform_(
            self.mix.weight, a=math.sqrt(5)
        )  # a是负斜率，ReLU默认使用的是0
        # 正交初始化
        # torch.nn.init.orthogonal_(self.mix.weight)

        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base):
        # print('base shape', base.shape)
        rotated_base = self.rotate_layer(base)
        # print( self.learned_source(base).dtype )
        output = base + torch.matmul(
            self.mix(self.act_fn(self.learned_source(base)) - rotated_base),
            self.rotate_layer.weight.T,
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True
        ).to(self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[
            :, :overload_w_width
        ] = overload_w
        assert (
            torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True
        )  # we must match!

        return


class RedIntervention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.delta_vector = nn.ParameterDict(
            {
                "activation_scaling": nn.Parameter(torch.ones(1, self.embed_dim)),
                "activation_bias": nn.Parameter(torch.zeros(1, self.embed_dim)),
            }
        )

        # # 通过修改均值和方差调整初始化
        # activation_bias_mean = 1e-4
        # activation_bias_var = 1e-2
        # activation_scaling_mean = 9e-1
        # activation_scaling_var = 3e-2

        # # 根据方差修改初始化值的尺度
        # activation_bias_std = torch.sqrt(torch.tensor(activation_bias_var))
        # activation_scaling_std = torch.sqrt(torch.tensor(activation_scaling_var))

        # # 使用均值和方差初始化
        # self.delta_vector = nn.ParameterDict({
        #     "activation_bias": nn.Parameter(torch.empty(1, self.embed_dim)),
        #     "activation_scaling": nn.Parameter(torch.empty(1, self.embed_dim)),
        # })

        # # torch.nn.init.normal_(tensor, mean=0, std=1)
        # nn.init.normal_(self.delta_vector["activation_bias"], mean=activation_bias_mean, std=activation_bias_std)
        # nn.init.normal_(self.delta_vector["activation_scaling"], mean=activation_scaling_mean, std=activation_scaling_std)

    def forward(self, base):
        hidden_states = base * self.delta_vector["activation_scaling"]
        hidden_states = hidden_states + self.delta_vector["activation_bias"]
        return hidden_states.to(base.dtype)


class DualSpacesIntervention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        r = kwargs["low_rank_dimension"] * 2
        k = kwargs["low_rank_dimension"]
        self.lora_A = nn.Parameter(torch.zeros((k, self.embed_dim)))
        self.lora_E = nn.Parameter(torch.zeros(r, 1))
        self.lora_B = nn.Parameter(torch.zeros((r, k)))
        self.lora_aa = nn.Parameter(torch.zeros((k, r)))
        self.lora_bb = nn.Parameter(torch.zeros((self.embed_dim, k)))
        self.reset_parameters()

    def forward(self, base):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        result1 = self.lora_A.T @ self.lora_B.T
        result2 = self.lora_aa.T @ self.lora_bb.T
        lora_E_flat = self.lora_E.view(-1)
        lora_E_diag = torch.diag(lora_E_flat)
        result = base + base @ (result1 @ lora_E_diag @ result2).to(base.dtype)
        return result.to(base.dtype)

    def reset_parameters(self):
        nn.init.zeros_(self.lora_E)
        nn.init.orthogonal_(self.lora_A)
        nn.init.orthogonal_(self.lora_B)
        nn.init.orthogonal_(self.lora_aa)
        nn.init.orthogonal_(self.lora_bb)


class LoRAIntervention(nn.Module):
    """
    LoRA modification: h + W_LoRA * (h - W_LoRA^T * h)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, keep_last_dim=True)

        self.rank = kwargs["low_rank_dimension"]
        self.embed_dim = (
            self.embed_dim
        )  # Assuming embed_dim is defined in the parent class

        # Low-rank matrices for LoRA
        # self.lora_A = torch.nn.Linear(self.embed_dim, self.rank, bias=False)
        # self.lora_B = torch.nn.Linear(self.rank, self.embed_dim, bias=False)
        # Initialize parameters (you can also set a specific initialization here)
        # torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=0.1)
        # torch.nn.init.kaiming_uniform_(self.lora_B.weight, a=0.1)
        # self.lora_A = nn.Parameter(self.rank, self.embed_dim)
        # self.lora_B = nn.Parameter(self.embed_dim, self.rank)
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.embed_dim, dtype=torch.bfloat16)
        )
        self.lora_B = nn.Parameter(
            torch.empty(self.embed_dim, self.rank, dtype=torch.bfloat16)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )
        self.scaling = 1.0

    def forward(
        self,
        base: torch.Tensor,
        source=None,
        subspaces=None,
    ) -> torch.Tensor:
        # Compute the LoRA output
        # lora_output = self.lora_B(self.act_fn(self.lora_A(base)))
        # output = base + lora_output
        # return self.dropout(output.to(base.dtype))
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = (
            base
            + (
                self.dropout(base)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            )
            * self.scaling
        )
        return result.to(base.dtype)

    def state_dict(self, *args, **kwargs) -> OrderedDict:
        """Overwrite for data-efficiency."""
        state_dict = OrderedDict()
        # state_dict.update(self.lora_A.state_dict())
        # state_dict.update(self.lora_B.state_dict())

        state_dict.update(
            {
                "lora_A": self.lora_A.data,
                "lora_B": self.lora_B.data,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: dict, *args, **kwargs) -> None:
        """Overwrite for data-efficiency."""
        self.lora_A.load_state_dict(state_dict, strict=False)
        self.lora_B.load_state_dict(state_dict, strict=False)


class FeedForwardIntervention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.linear_1 = nn.Linear(self.embed_dim, self.embed_dim).to(torch.bfloat16)
        self.dropout = nn.Dropout(0.1)
        self.orthogonal_init()

    def orthogonal_init(self):
        # 获取线性层的权重
        weight = self.linear_1.weight.data
        # 转换为 float32 进行 QR 分解
        weight_float32 = weight.to(torch.float32)

        # 执行 QR 分解
        q, r = torch.qr(weight_float32)

        # 将正交化后的权重赋值回去
        self.linear_1.weight.data = q.to(torch.bfloat16)  # 确保保持数据类型

    def forward(
        self,
        base: torch.Tensor,
        source=None,
        subspaces=None,
    ):
        x = self.dropout(F.relu(self.linear_1(base)))
        return base + x.to(base.dtype)


# 多头注意力干预
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttentionIntervention(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, keep_last_dim=True)
        in_features = self.embed_dim
        head_num = 2
        bias = True
        activation = F.relu
        if in_features % head_num != 0:
            raise ValueError(
                "`in_features`({}) should be divisible by `head_num`({})".format(
                    in_features, head_num
                )
            )
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias).to(torch.bfloat16)
        self.linear_k = nn.Linear(in_features, in_features, bias).to(torch.bfloat16)
        self.linear_v = nn.Linear(in_features, in_features, bias).to(torch.bfloat16)
        self.linear_o = nn.Linear(in_features, in_features, bias).to(torch.bfloat16)

    def forward(
        self,
        base: torch.Tensor,
        source=None,
        subspaces=None,
    ):
        q, k, v = self.linear_q(base), self.linear_k(base), self.linear_v(base)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        mask = None
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y.to(base.dtype)

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return (
            torch.tril(torch.ones(seq_len, seq_len))
            .view(1, seq_len, seq_len)
            .repeat(batch_size, 1, 1)
        )

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def extra_repr(self):
        return "in_features={}, head_num={}, bias={}, activation={}".format(
            self.in_features,
            self.head_num,
            self.bias,
            self.activation,
        )


class NoreftIntervention(nn.Module):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class ConsreftIntervention(nn.Module):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True
        )

    def forward(self, base):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)


class LobireftIntervention(nn.Module):
    """
    LobiReFT(h) = h + R^T(b)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True
        )
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )

    def forward(self, base):
        output = base + torch.matmul(self.learned_source, self.rotate_layer.weight.T)
        return self.dropout(output.to(base.dtype))


class DireftIntervention(nn.Module):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base):
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(
                self.rotate_layer.weight.dtype
            ),
            self.rotate_layer.weight.T,
        )
        return self.dropout(output.to(base.dtype))


class NodireftIntervention(nn.Module):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base):
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


############################## moe related ################################

# constants
MIN_EXPERT_CAPACITY = 4


# helper functions
def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]


# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]


def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)


# activations


class GELU_(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


GELU = nn.GELU if hasattr(nn, "GELU") else GELU_

# expert class


class Experts(nn.Module):
    def __init__(self, dim, num_experts=16, hidden_dim=None, activation=GELU):
        super().__init__()

        # hidden_dim = default(hidden_dim, dim * 4)
        hidden_dim = 8
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim, dtype=torch.bfloat16)
        w2 = torch.zeros(*num_experts, hidden_dim, dim, dtype=torch.bfloat16)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum("...nd,...dh->...nh", x, self.w1)
        hidden = self.act(hidden)
        out = torch.einsum("...nh,...hd->...nd", hidden, self.w2)
        return out


# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network


class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps=1e-9,
        outer_expert_dims=tuple(),
        second_policy_train="random",
        second_policy_eval="random",
        second_threshold_train=0.2,
        second_threshold_eval=0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
    ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(
            torch.randn(*outer_expert_dims, dim, num_gates, dtype=torch.bfloat16)
        )

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance=None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum("...bnd,...de->...bne", x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.0).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1.0 - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.0).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates**2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0.0, 1.0)
            mask_2 *= (
                (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
            )
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(
            group_size, int((group_size * capacity_factor) / num_gates)
        )
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat

        # [batch, group, experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :]
            + gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        ).to(x.dtype)

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss


# sparse MoE
class MoEIntervention(nn.Module):
    def __init__(
        self,
        dim,
        num_experts=16,
        hidden_dim=None,
        activation=nn.ReLU,
        second_policy_train="random",
        second_policy_eval="random",
        second_threshold_train=0.2,
        second_threshold_eval=0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
        loss_coef=1e-2,
        experts=None,
        **kwargs,
    ):
        super().__init__(**kwargs, keep_last_dim=True)

        self.num_experts = num_experts

        gating_kwargs = {
            "second_policy_train": second_policy_train,
            "second_policy_eval": second_policy_eval,
            "second_threshold_train": second_threshold_train,
            "second_threshold_eval": second_threshold_eval,
            "capacity_factor_train": capacity_factor_train,
            "capacity_factor_eval": capacity_factor_eval,
        }
        self.gate = Top2Gating(dim, num_gates=num_experts, **gating_kwargs)
        self.experts = default(
            experts,
            lambda: Experts(
                dim,
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                activation=activation,
            ),
        )
        self.loss_coef = loss_coef

    def forward(self, inputs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum("bnd,bnec->ebcd", inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = inputs + torch.einsum(
            "ebcd,bnec->bnd", expert_outputs, combine_tensor
        ).to(inputs.dtype)
        # return output, loss * self.loss_coef
        return output
