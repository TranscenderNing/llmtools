from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq)


def load_representation_model(
    model_path,
    # peft config
    lora_n=4,
    lora_r=2,
    lora_alpha=16,
    lora_target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    mode="melora",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
    )
    get peft model
    config = MELoraConfig(
        r=[lora_r] * lora_n,
        lora_alpha=[lora_alpha] * lora_n,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        mode=mode,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model