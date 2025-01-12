import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import uuid
import torch
from transformers import (
    AutoConfig,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    set_seed,
    TrainingArguments,
    TrainerCallback,
)

import json

from utils import (
    get_args,
    dtype_mapping,
    load_tokenizer,
    task_config,
    load_peft_model,
    load_base_model,
    compute_metrics,
)
from tier_dataset import NormalPeftNLGDataset
from tier_trainer import TierTrainer
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"





class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            cached = torch.cuda.memory_reserved() / 1024 ** 2
            print(f"Step {state.global_step}: Allocated {allocated:.2f} MB, Cached {cached:.2f} MB")

memory_callback = MemoryCallback()


def train(args):
    dtype = dtype_mapping[args.dtype]
    set_seed(args.seed)

    model_config = AutoConfig.from_pretrained(args.model_path)
    print(model_config)
    run_name = (
        f"{model_config._name_or_path.replace('/','-')}.{args.task}.{uuid.uuid1()}"
    )
    print(f"args: {args}")
    print(f"model info: {model_config}")
    print(f"run_name: {run_name}")

    # load tokenizer
    tokenizer, need_resize = load_tokenizer(args.model_path, max_length=args.max_length)

    train_datasets_arr = task_config[args.task]["train_datasets"]
    eval_datasets_arr = task_config[args.task]["eval_datasets"]

    train_dataset = NormalPeftNLGDataset(
        args.task,
        os.path.join(args.data_dir, train_datasets_arr[0]),
        tokenizer,
        data_split="train",
        seed=args.seed,
        max_n_example=args.max_n_train_example,
        task_config=task_config,
        method=args.peft_method,
        model_name=args.model_name
    )
    trigger_tokens = train_dataset.trigger_tokens

    print("train ds 0: ", train_dataset[0])

    # load model based on task type.
    model, model_config = load_base_model(
        args.model_path, dtype, device, len(tokenizer), need_resize=need_resize
    )
    
    peft_model = load_peft_model(model, peft_method=args.peft_method)
    print(f"peft model:{peft_model}")

    # disable origianl model gradients
    peft_model.print_trainable_parameters()


    # train related
    # select collator based on the type
    if args.task == "glue":
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest"
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
        )
    data_collator = data_collator_fn

    # training args
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch" if args.task == "glue" else "no",
        save_strategy="epoch" if args.task == "glue" else "no",
        metric_for_best_model=(
            args.metric_for_best_model if args.task == "glue" else None
        ),
        load_best_model_at_end=True if args.task == "glue" else False,
        logging_strategy="steps",
        save_total_limit=1,  # for GLUE, it will save 2 at max.
        logging_steps=args.logging_steps,
        lr_scheduler_type=args.schedule,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        report_to="wandb" if args.is_wandb else "none",
        use_cpu=False if device == "cuda" else True,
        seed=args.seed,
        # until HF supports ReFT, this remains False! :)
        remove_unused_columns=True,
    )
    trainer = TierTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[memory_callback]
    )
    trainer.train()
    
    print("训练结束")
    exit(0)

    # saving final model and hyperparameters
    print("Saving final model")
    peft_model.save_pretrained(f"{args.output_dir}/{run_name}")
    print(f"final model is saved in {args.output_dir}/{run_name}")
    args_dict = vars(args)

    json_file_name = f"{args.output_dir}/{run_name}/args.json"
    with open(json_file_name, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)
    print(f"hyperparameters are saved in {args.output_dir}/{run_name}/args.json")

    # eval
    peft_model.model.eval()

    all_eval_datasets = {}
    for eval_dataset in eval_datasets_arr:
        all_eval_datasets[eval_dataset] = {}
        raw_eval = NormalPeftNLGDataset(
            args.task,
            os.path.join(args.data_dir, eval_dataset),
            tokenizer,
            data_split=args.test_split,
            seed=args.seed,
            max_n_example=args.max_n_eval_example,
            task_config=task_config,
            method=args.peft_method,
            model_name=args.model_name
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
                peft_model,
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
                f"{args.output_dir}/{run_name}/{dataset_name}_{split}_outputs.json"
            )
            with open(result_json_file_name, "w") as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{args.output_dir}/{run_name}/eval_results.json"
    with open(result_json_file_name, "w") as json_file:
        json.dump(eval_results, json_file, indent=4)
    print(
        f"Training and evalating results can be found in {args.output_dir}/{run_name}"
    )


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
