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
)

import json

from utils import (
    get_pred_args,
    dtype_mapping,
    load_tokenizer,
    task_config,
    intervention_mapping,
    load_base_model,
    TierDataCollator,
    compute_metrics,
)
from tier_dataset import TierNLGDataset
from tier_config import TierConfig
from tier_model import TierModel
from tier_trainer import TierTrainer
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(args):
    dtype = dtype_mapping[args.dtype]
    set_seed(args.seed)

    model_config = AutoConfig.from_pretrained(args.model_path)
    print(model_config)
    run_name = (
        f"{model_config._name_or_path.replace('/','-')}.{args.task}.{uuid.uuid1()}"
    )
    os.makedirs(f"{args.output_dir}/{run_name}", exist_ok=True)
    print(f"args: {args}")
    print(f"model info: {model_config}")
    print(f"run_name: {run_name}")

    # load tokenizer
    tokenizer, need_resize = load_tokenizer(args.model_path, max_length=args.max_length)
    eval_datasets_arr = task_config[args.task]["eval_datasets"]

    # load model based on task type.
    model, model_config = load_base_model(
        args.model_path, dtype, device, len(tokenizer), need_resize=need_resize
    )
    # get tier model
    tier_model = TierModel.from_pretrained(args.tier_model_path, model)

    # select collator based on the type
    if args.task == "glue":
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest"
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
        )
    data_collator = TierDataCollator(data_collator=data_collator_fn)

    # eval
    tier_model.model.eval()
    for k, v in tier_model.interventions.items():
        _ = v[0].eval()
    all_eval_datasets = {}
    print(
        "num_interventions",
        tier_model.num_interventions,
        "position",
        tier_model.position,
    )
    for eval_dataset in eval_datasets_arr:
        all_eval_datasets[eval_dataset] = {}
        raw_eval = TierNLGDataset(
            args.task,
            os.path.join(args.data_dir, eval_dataset),
            tokenizer,
            data_split=args.test_split,
            seed=args.seed,
            max_n_example=args.max_n_eval_example,
            num_interventions=tier_model.num_interventions,
            position=tier_model.position,
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
                tier_model,
                tokenizer,
                eval_dataset,
                data_items,
                task_config[args.task]["trigger_tokens"],
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
    print(f"evalating results can be found in {args.output_dir}/{run_name}")


def main():
    args = get_pred_args()
    train(args)


if __name__ == "__main__":
    main()
