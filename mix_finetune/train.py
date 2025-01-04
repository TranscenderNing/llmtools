import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import uuid
from transformers import AutoTokenizer, TrainingArguments
import os
import torch
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq
from model import RepresentationLLama
from task_config import task_config
from dataset import CommonSupervisedDataset, CommonDataCollator
from compute_metrics import compute_common_metrics
from mix_trainer import CommonTrainer
from peft import MELoraConfig, get_peft_model
from transformers.trainer_callback import TrainerCallback

device_map = "cuda"


def load_mix_model(
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
    # get peft model
    # config = MELoraConfig(
    #     r=[lora_r] * lora_n,
    #     lora_alpha=[lora_alpha] * lora_n,
    #     target_modules=lora_target_modules,
    #     lora_dropout=lora_dropout,
    #     bias="none",
    #     mode=mode,
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, config)
    model = RepresentationLLama(model, op_position='post_attention_layernorm')
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

class CustomModelSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_path = os.path.join(checkpoint_path)

        kwargs["model"].save_model(save_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path) :
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
        if  "model.safetensors" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "model.safetensors"))



def train(
    model_path: str = "",
    data_dir: str = "",
    output_dir: str = "",
    # learning_rate: float = 2e-5,
    learning_rate: float = 3e-4,
    num_train_epochs: int = 0.2,
    task: str = "commonsense",
    seed: int = 42,
    max_n_train_example=None,
    max_n_eval_example=None,
    test_split="test",
    max_length=2048,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    weight_decay=0.00,
    greedy_decoding=True,
    temperature=None,
    top_p=None,
    top_k=None,
):

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, model_max_length=max_length,)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    # process dataset
    assert task in task_config, f"Unrecognized task: {task}"
    train_datasets = task_config[task]["train_datasets"]
    eval_datasets = task_config[task]["eval_datasets"]
    print("train datasets is", train_datasets)
    print("eval datasets is", eval_datasets)

    train_dataset = CommonSupervisedDataset(
        task,
        os.path.join(data_dir, train_datasets[0]),
        tokenizer,
        data_split="train",
        seed=seed,
        max_n_example=max_n_train_example,
    )
    trigger_tokens = train_dataset.trigger_tokens
    print("train ds 0: ", train_dataset[0])

    all_eval_datasets = {}
    for eval_dataset in eval_datasets:
        test_splits = test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = CommonSupervisedDataset(
                task,
                os.path.join(data_dir, eval_dataset),
                tokenizer,
                data_split=split,
                seed=seed,
                max_n_example=max_n_eval_example,
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets
    
    # exit(0)
    # load model
    model = load_mix_model(model_path=model_path)
    
    # model keylist and sub module
    # key_list = [key for key, _ in model.named_modules()]
    # for key in key_list:
    #     print(key)
    # sub_module = model.base_model.get_sub_module(key_list[0])
    # print(sub_module)
    # exit(0)
    
    trainable_params = count_parameters(model)
    total_params = count_total_parameters(model)
    trainable_params_ratio = trainable_params / total_params
    # 输出
    print(f'Total number of trainable parameters: {trainable_params}')
    print(f'Total number of parameters: {total_params}')
    print(f'Ratio of trainable parameters to total parameters: {trainable_params_ratio:.4f}')

    print("mix model is", model)
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    )
    data_collator = CommonDataCollator(data_collator=data_collator_fn)

    # training
    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model=None,
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=1,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        weight_decay=weight_decay,
        report_to="none",
        use_cpu=False,
        seed=seed,
        remove_unused_columns=False,
    )
    # make trainer
    trainer = CommonTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[CustomModelSavingCallback()]
    )
    trainer.train()

    # saving the final model
    print("Saving model...")
    output_dir = os.path.join(output_dir, "final_checkpoint")
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(output_dir)

    # do evaluate
    model.eval()
    eval_results = {}
    for dataset_name in eval_datasets:
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():
            generations, stats = compute_common_metrics(
                task,
                dataset_name,
                model,
                tokenizer,
                eval_dataset,
                data_items,
                trigger_tokens,
                per_device_eval_batch_size,
                data_collator=None,
                split=split,
                greedy_decoding=greedy_decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # log
            eval_results.update(stats)

            generations = stats if generations is None else generations
            result_json_file_name = f"{output_dir}/{dataset_name}_{split}_outputs.json"
            with open(result_json_file_name, "w") as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{output_dir}/eval_results.json"
    with open(result_json_file_name, "w") as json_file:
        json.dump(eval_results, json_file, indent=4)
    print(f"Training results can be found in {output_dir}")


def main():
    model_path = "/home/ldn/.cache/huggingface/hub/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75"
    data_path = "/home/ldn/baidu/pyreft/paddle-version/loreft/datasets"
    output_dir = "/home/ldn/baidu/reft-pytorch-codes/mix_finetune/train-outputs/"
    unique_id = uuid.uuid4().hex  # 生成一个32位的16进制字符串
    output_dir = os.path.join(output_dir, unique_id)
    os.makedirs(output_dir, exist_ok=True)

    train(
        model_path=model_path,
        data_dir=data_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
