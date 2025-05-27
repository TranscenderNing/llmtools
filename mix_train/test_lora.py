from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 加载预训练模型
model_name = "/data/ldn/llm-models/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# 确定总层数和起始层索引
num_layers = len(model.model.layers)  # 假设模型有32层
start_layer = num_layers - 16  # 后面16层的起始索引（如16到31层）

# 生成目标模块列表（假设每层包含q_proj和v_proj）
target_modules = [
    f"model.layers.{i}.self_attn.q_proj"
    for i in range(start_layer, num_layers)
] + [
    f"model.layers.{i}.self_attn.v_proj"
    for i in range(start_layer, num_layers)
]

# 配置LoRA，仅作用于目标模块
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA适配器
model = get_peft_model(model, peft_config)
print(model)
model.print_trainable_parameters()  # 查看可训练参数数量