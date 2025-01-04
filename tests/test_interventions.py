import math
import torch
import torch.nn as nn
import torch.nn.functional as F



mora256 = {
    "eval/boolq": 0.6474006116207951,
    "eval/piqa": 0.8106637649619152,
    "eval/social_i_qa": 0.785568065506653,
    "eval/hellaswag": 0.9121688906592312,
    "eval/winogrande": 0.7734806629834254,
    "eval/ARC-Easy": 0.8143939393939394,
    "eval/ARC-Challenge": 0.6331058020477816,
    "eval/openbookqa": 0.784,
    # "n_params": 2171136
}

dualspace = {
    "eval/boolq": 0.6938837920489297,
    "eval/piqa": 0.8307943416757345,
    "eval/social_i_qa": 0.8039918116683725,
    "eval/hellaswag": 0.9204341764588727,
    "eval/winogrande": 0.8263614838200474,
    "eval/ARC-Easy": 0.8253367003367004,
    "eval/ARC-Challenge": 0.6578498293515358,
    "eval/openbookqa": 0.778,
    # "n_params": 2171664
}




alternate_inter = {
    "eval/boolq": 0.6993883792048929,
    "eval/piqa": 0.824265505984766,
    "eval/social_i_qa": 0.797338792221085,
    "eval/hellaswag": 0.9318860784704243,
    "eval/winogrande": 0.8216258879242304,
    "eval/ARC-Easy": 0.8291245791245792,
    "eval/ARC-Challenge": 0.6919795221843004,
    "eval/openbookqa": 0.792,
    # "n_params": 16910336
}


melora = {
    "eval/boolq": 0.7165137614678899,
    "eval/piqa": 0.8498367791077258,
    "eval/social_i_qa": 0.8014329580348004,
    "eval/hellaswag": 0.9342760406293567,
    "eval/winogrande": 0.8342541436464088,
    "eval/ARC-Easy": 0.8446969696969697,
    "eval/ARC-Challenge": 0.7175767918088737,
    "eval/openbookqa": 0.832
}


peft_reft_mix = {
    "eval/boolq": 0.6241590214067279,
    "eval/piqa": 0.4940152339499456,
    "eval/social_i_qa": 0.34698055271238487,
    "eval/hellaswag": 0.26687910774746065,
    "eval/winogrande": 0.49329123914759276,
    "eval/ARC-Easy": 0.2563131313131313,
    "eval/ARC-Challenge": 0.25853242320819114,
    "eval/openbookqa": 0.272,
    # "n_params": 131072
}


peft_reft_mix_no_ga = {
    "eval/boolq": 0.6155963302752293,
    "eval/piqa": 0.5919477693144722,
    "eval/social_i_qa": 0.4524053224155578,
    "eval/hellaswag": 0.3855805616411073,
    "eval/winogrande": 0.4909234411996843,
    "eval/ARC-Easy": 0.4116161616161616,
    "eval/ARC-Challenge": 0.34044368600682595,
    "eval/openbookqa": 0.306,
    # "n_params": 131072
}




peft_reft_lr = {
    "eval/boolq": 0.6529051987767585,
    "eval/piqa": 0.8035908596300326,
    "eval/social_i_qa": 0.7538382804503583,
    "eval/hellaswag": 0.8984266082453695,
    "eval/winogrande": 0.7687450670876085,
    "eval/ARC-Easy": 0.8009259259259259,
    "eval/ARC-Challenge": 0.6006825938566553,
    "eval/openbookqa": 0.694,
    # "n_params": 131072
}




peft_reft_melora = {
    "eval/boolq": 0.692354740061162,
    "eval/piqa": 0.8177366702937976,
    "eval/social_i_qa": 0.7932446264073695,
    "eval/hellaswag": 0.9103764190400319,
    "eval/winogrande": 0.8208366219415943,
    "eval/ARC-Easy": 0.8164983164983165,
    "eval/ARC-Challenge": 0.6706484641638225,
    "eval/openbookqa": 0.784,
    # "n_params": 8192
}



melora_plus = {
    "eval/boolq": 0.6871559633027523,
    "eval/piqa": 0.8117519042437432,
    "eval/social_i_qa": 0.7871033776867963,
    "eval/hellaswag": 0.9045010953993229,
    "eval/winogrande": 0.8011049723756906,
    "eval/ARC-Easy": 0.8118686868686869,
    "eval/ARC-Challenge": 0.6322525597269625,
    "eval/openbookqa": 0.78
}

red_melora = {
    "eval/boolq": 0.671559633027523,
    "eval/piqa": 0.8046789989118607,
    "eval/social_i_qa": 0.7548618219037871,
    "eval/hellaswag": 0.8773152758414658,
    "eval/winogrande": 0.7569060773480663,
    "eval/ARC-Easy": 0.7988215488215489,
    "eval/ARC-Challenge": 0.6262798634812287,
    "eval/openbookqa": 0.722
}


loref_melora = {
    "eval/boolq": 0.6868501529051988,
    "eval/piqa": 0.8177366702937976,
    "eval/social_i_qa": 0.793756397134084,
    "eval/hellaswag": 0.9147580163314081,
    "eval/winogrande": 0.8208366219415943,
    "eval/ARC-Easy": 0.8097643097643098,
    "eval/ARC-Challenge": 0.6655290102389079,
    "eval/openbookqa": 0.796,
    # "n_params": 2097408
}



orth_melora ={
    "eval/boolq": 0.7021406727828746,
    "eval/piqa": 0.8318824809575626,
    "eval/social_i_qa": 0.7993858751279427,
    "eval/hellaswag": 0.9277036446922924,
    "eval/winogrande": 0.8216258879242304,
    "eval/ARC-Easy": 0.8341750841750841,
    "eval/ARC-Challenge": 0.6800341296928327,
    "eval/openbookqa": 0.812
}



mevera = {
    "eval/boolq": 0.6801223241590214,
    "eval/piqa": 0.8117519042437432,
    "eval/social_i_qa": 0.7794268167860798,
    "eval/hellaswag": 0.9004182433778132,
    "eval/winogrande": 0.7845303867403315,
    "eval/ARC-Easy": 0.8186026936026936,
    "eval/ARC-Challenge": 0.6535836177474402,
    "eval/openbookqa": 0.748
}


mevera_b = {
    "eval/boolq": 0.6825688073394496,
    "eval/piqa": 0.8133841131664853,
    "eval/social_i_qa": 0.781985670419652,
    "eval/hellaswag": 0.9017128062139016,
    "eval/winogrande": 0.7932123125493291,
    "eval/ARC-Easy": 0.8253367003367004,
    "eval/ARC-Challenge": 0.6638225255972696,
    "eval/openbookqa": 0.754
}

mevera_all = {
    "eval/boolq": 0.7091743119266055,
    "eval/piqa": 0.8269858541893362,
    "eval/social_i_qa": 0.7998976458546572,
    "eval/hellaswag": 0.9271061541525593,
    "eval/winogrande": 0.8255722178374112,
    "eval/ARC-Easy": 0.8375420875420876,
    "eval/ARC-Challenge": 0.6766211604095563,
    "eval/openbookqa": 0.802
}

def get_latex_str(acc_dict):
    sum = 0
    latex_str = []
    for key, val in acc_dict.items():
        print(key,val)
        num = round(val, 4) * 100
        latex_str.append(str(num))
        sum += num
        

    avg = round(sum / len(acc_dict), 2)

    latex_str.append(str(avg))
    print(' & '.join(latex_str))
    return 


    
c01 = {
    "eval/boolq": 0.6608562691131499,
    "eval/piqa": 0.7959738846572362,
    "eval/social_i_qa": 0.7563971340839304,
    "eval/hellaswag": 0.8749253136825333,
    "eval/winogrande": 0.7466456195737964,
    "eval/ARC-Easy": 0.8076599326599326,
    "eval/ARC-Challenge": 0.6271331058020477,
    "eval/openbookqa": 0.692
}

c001 = {
    "eval/boolq": 0.6602446483180429,
    "eval/piqa": 0.79379760609358,
    "eval/social_i_qa": 0.7579324462640737,
    "eval/hellaswag": 0.8749253136825333,
    "eval/winogrande": 0.7482241515390686,
    "eval/ARC-Easy": 0.8068181818181818,
    "eval/ARC-Challenge": 0.6245733788395904,
    "eval/openbookqa": 0.692
}

c0001 = {
    "eval/boolq": 0.6587155963302752,
    "eval/piqa": 0.7916213275299239,
    "eval/social_i_qa": 0.7548618219037871,
    "eval/hellaswag": 0.874726150169289,
    "eval/winogrande": 0.7490134175217048,
    "eval/ARC-Easy": 0.8042929292929293,
    "eval/ARC-Challenge": 0.6305460750853242,
    "eval/openbookqa": 0.692
}



get_latex_str(c01)
get_latex_str(c001)
get_latex_str(c0001)










exit(0)






class MloraIntervention(
    nn.Module
):
    def __init__(self, embed_dim):
        super(MloraIntervention, self).__init__()
        self.r =  128
        self.embed_dim = embed_dim
        self.learned_source = torch.nn.Linear(
            self.embed_dim, 128
        ).to( torch.bfloat16)
        self.dropout = torch.nn.Dropout(
         0.0
        )

    def forward(self, x, source=None, subspaces=None):
        in_f = out_f =  self.embed_dim
        r = self.r
        sum_inter = in_f // r
        rb1 = in_f//r if in_f % r == 0 else in_f//r + 1
        if in_f % r != 0:
            pad_size = r - in_f % r
            x = torch.cat([x, x[..., :pad_size]], dim=-1)
            sum_inter += 1
        in_x = x.view(*x.shape[:-1], sum_inter, r)
        if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, r, 2).float() / r))
            t = torch.arange(rb1)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
            self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
        rh_in_x = torch.cat((-in_x[..., r//2:], in_x[..., :r//2]), dim=-1)
        in_x = in_x*self.cos + rh_in_x*self.sin
        
        # print('in_x.shape is', in_x.shape)
        # print('self.learned_source shape is', self.learned_source.weight.shape)
        out_x = self.learned_source(in_x)
        out_x = out_x.view(*x.shape[:-1], -1)[..., :out_f]

        output = x + out_x
        return self.dropout(output.to(x.dtype))

class RedIntervention(
    nn.Module
):
    """
    LoReFT(h) = h + R^T W' (Wh + b − Rh)
    """

    def __init__(self, embed_dim):
        super(RedIntervention, self).__init__()
        self.embed_dim = embed_dim
        self.delta_vector = nn.ParameterDict({
            "activation_scaling": nn.Parameter(torch.ones(1, self.embed_dim)),
            "activation_bias":nn.Parameter(torch.zeros(1, self.embed_dim)),
        })

    def forward(self, base, source=None, subspaces=None):
        hidden_states = base * self.delta_vector["activation_scaling"]
        hidden_states = hidden_states + self.delta_vector["activation_bias"]
        return hidden_states.to(base.dtype)

# 测试  
def test_red_intervention():  
    # 假设 embed_dim 为 5  
    embed_dim = 4096  
    model = MloraIntervention(embed_dim=4096)  
  
    # 创建一个随机的 base 输入张量，形状为 (batch_size, seq_len, embed_dim)  
    batch_size = 2  
    seq_len = 3  
    base = torch.randn(batch_size, seq_len, embed_dim).to(torch.bfloat16) 
  
    # 打印输入  
    print("Input (base):")  
    print(base.shape)  
  
    # 前向传播  
    output = model(base)  
  
    # 打印输出  
    print("Output:")  
    print(output.shape)  
  
    # 预期输出应该与 base 形状相同，且由于 activation_scaling 初始化为 1，activation_bias 初始化为 0，  
    # 输出应该与 base 非常接近（除了可能的数值误差）  
    # assert output.shape == base.shape, "Output shape does not match input shape"  
    # assert torch.allclose(output, base, atol=1e-6), "Output does not match expected transformation"  
  
# 运行测试  
test_red_intervention()

exit(0)


__all__ = ["MultiHeadAttention", "ScaledDotProductAttention"]


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self, in_features, head_num, bias=True, activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
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
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

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


import torch

# 假设输入特征的大小为64，头的数量为8，批次大小为10，序列长度为20
in_features = 32
head_num = 4
batch_size = 2
seq_len = 10

# 初始化多头注意力层
multi_head_attention = MultiHeadAttention(in_features=in_features, head_num=head_num)

# 随机生成查询、键和值的张量
q = torch.rand(batch_size, seq_len, in_features)  # 查询
k = torch.rand(batch_size, seq_len, in_features)  # 键
v = torch.rand(batch_size, seq_len, in_features)  # 值

# 可选：生成一个mask，假设我们要屏蔽某些位置
mask = MultiHeadAttention.gen_history_mask(q)  # 生成历史mask

# 前向传播
output = multi_head_attention(q, k, v, mask)

print(
    "Output shape:", output.shape
)  # 输出形状应该为 (batch_size, seq_len, in_features)
