
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    DataCollatorForSeq2Seq,
)

from datasets import Dataset





def make_dataloader(
    dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn
    )





class CommonTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False
    ):
        base_outputs = model(
            **{
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            labels=inputs["labels"],
        )
        return base_outputs.loss
    
    # 配置不同的学习率
    # def create_optimizer(self):
    #     reft_lr = self.args.learning_rate
    #     lora_lr = 3e-4
    #     print('reft lr: {}, lora lr: {}'.format(reft_lr, lora_lr))
    #     opt_model = self.model.model
    #     if self.optimizer is None:
    #         param_groups = {
    #             "lora": {},
    #             "interventions_net": {},
    #         }
    #         # reft params
    #         for k, v in self.model.interventions.items():
    #             for name, param in v[0].named_parameters():
    #                 param_groups["interventions_net"][k + name] = param

    #         for name, param in opt_model.named_parameters():
    #             if "lora_" in name and param.requires_grad:
    #                 param_groups["lora"][name] = param

    #         print('lora parameters:', param_groups["lora"])
    #         print('interventions_net parameters:', param_groups["interventions_net"])

            
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": list(param_groups["interventions_net"].values()),
    #                 "weight_decay": self.args.weight_decay,
    #                 "lr": reft_lr
    #             },
    #             {
    #                 "params": list(param_groups["lora"].values()),
    #                 "weight_decay": self.args.weight_decay,
    #                 "lr": lora_lr
    #             }
    #         ]

    #         optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    #         self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
    #     return self.optimizer

