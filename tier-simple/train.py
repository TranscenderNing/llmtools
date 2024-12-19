import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import os
import uuid

import torch
from dataset import NLGDataset
from peft import MELoraConfig, get_peft_model
from tier_config import TierConfig
from tier_model import RepresentationModel
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from transformers.trainer_callback import TrainerCallback
from utils import (compute_metrics, dtype_mapping, get_args,
                   load_representation_model, load_tokenizer, task_config)

device = "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train(args):
    dtype = dtype_mapping[args.dtype]
    model_config = AutoConfig.from_pretrained(args.model_path)
    print(model_config)
    # load tokenizer
    tokenizer, need_resize = load_tokenizer(args.model_path, max_length=512)

    train_datasets_arr = task_config[args.task]["train_datasets"]
    eval_datasets_arr = task_config[args.task]["eval_datasets"]

    train_dataset = NLGDataset(
        args.task,
        os.path.join(args.data_dir, train_datasets_arr[0]),
        tokenizer,
        data_split="train",
        seed=args.seed,
        max_n_example=args.max_n_train_example,
        task_config=task_config,
    )
    trigger_tokens = train_dataset.trigger_tokens

    print("train ds 0: ", train_dataset[0])
    # load model
    tier_config = TierConfig(
        op_position="post_attention_layernorm",
        intervention_type="red",
        intervention_params={
            "hidden_size": model_config.hidden_size,
            "dtype": dtype_mapping[args.dtype],
        },
    )
    model = load_representation_model(
        args.model_path,
        dtype,
        device,
        len(tokenizer),
        need_resize=need_resize,
        tier_config=tier_config,
    )
    print("model: ", model)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    )
    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model=None,
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=1,
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        report_to="none",
        use_cpu=False,
        seed=args.seed,
        remove_unused_columns=True,
    )
    # make trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )
    trainer.train()

    # saving the final model
    print("Saving fianl model...")
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(output_dir)

    # do evaluate
    model.eval()

    all_eval_datasets = {}
    for eval_dataset in eval_datasets_arr:
        all_eval_datasets[eval_dataset] = {}
        raw_eval = NLGDataset(
            args.task,
            os.path.join(args.data_dir, eval_dataset),
            tokenizer,
            data_split=args.test_split,
            seed=args.seed,
            max_n_example=args.max_n_eval_example,
            task_config=task_config,
        )
        all_eval_datasets[eval_dataset][args.test_split] = [
            raw_eval,
            raw_eval.raw_dataset,
        ]

    eval_results = {}
    for dataset_name in all_eval_datasets:
        for split, (eval_dataset, data_items) in all_eval_datasets[
            dataset_name
        ].items():
            generations, stats = compute_metrics(
                args.task,
                dataset_name,
                model,
                tokenizer,
                eval_dataset,
                data_items,
                trigger_tokens,
                args.eval_batch_size,
                data_collator if args.task == "gule" else data_collator,
                args.greedy_decoding,
                args.temperature,
                args.top_p,
                args.top_k,
                device=device,
            )

            # save predictions
            eval_results.update(stats)
            generations = stats if generations is None else generations
            result_json_file_name = (
                f"{args.output_dir}/{dataset_name}_{split}_outputs.json"
            )
            with open(result_json_file_name, "w") as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{args.output_dir}/eval_results.json"
    with open(result_json_file_name, "w") as json_file:
        json.dump(eval_results, json_file, indent=4)
    print(f"Training and evalating results can be found in {args.output_dir}")


def main():
    unique_id = uuid.uuid4().hex  # 生成一个32位的16进制字符串
    args = get_args()
    args.output_dir = os.path.join(args.output_dir, unique_id)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
