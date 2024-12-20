from utils import load_model_and_tokenizer, topk_intermediate_confidence_heatmap
import torch

class LlmExplanation:
    def __init__(self, model_path, load_reft_model=False, device_map=""):
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, device_map=device_map, load_reft_model=load_reft_model)
        if load_reft_model:
            self.layer_nums = 32
            self.layer_sums = 33
        else:
            self.layer_nums = self.model.config.num_hidden_layers
            self.layer_sums = self.layer_nums + 1
        self.forward_info = {}

    def get_forward_info(self, dataset):
        for _, i in enumerate(dataset):
            list_hs, tl_pair = self.step_forward_normal(self.model, self.tokenizer, i)
            last_hs = [hs[:, -1, :] for hs in list_hs]
            self.forward_info[_] = {"hidden_states": last_hs, "top-value_pair": tl_pair}

    # 每一层的embedding直接用lm_head解码
    def step_forward_normal(self, model, tokenizer, prompt, decoding=True, k_indices=5):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs.to(model.device)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            outputs = model(input_ids)
            tl_pair = []
            lm_head = model.lm_head
            if hasattr(model, "model") and hasattr(model.model, "norm"):
                norm = model.model.norm
            elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                norm = model.transformer.ln_f
            else:
                raise ValueError(f"Incorrect Model")
            for i, r in enumerate(outputs.hidden_states):
                layer_logits = []
                layer_output = norm(r)
                logits = lm_head(layer_output)
                next_token_logits = logits[:, -1, :]
                top_logits_k = k_indices
                top_values, top_indices = torch.topk(next_token_logits, top_logits_k, dim=-1)
                decoded_texts = [tokenizer.decode([idx], skip_special_tokens=False) for idx in top_indices.squeeze().tolist()]
                top_values = top_values.detach().cpu()
                if decoding:
                    for value, token in zip(top_values.squeeze().tolist(), decoded_texts):
                        layer_logits.append([token, value])
                else:
                    for value, token in zip(top_values.squeeze().tolist(), top_indices.squeeze().tolist()):
                        layer_logits.append([token, value])
                tl_pair.append(layer_logits)
        res_hidden_states = []
        for _ in outputs.hidden_states:
            res_hidden_states.append(_.float().detach().cpu().numpy())
        return res_hidden_states, tl_pair



    # 每一层的embedding直接用lm_head解码
    def step_forward_reft(self, model, tokenizer, prompt, decoding=True, k_indices=5):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(inputs)
            tl_pair = []
            lm_head = model.model.lm_head
            if hasattr(model.model, "model") and hasattr(model.model.model, "norm"):
                norm = model.model.model.norm
            elif hasattr(model.model, "transformer") and hasattr(model.transformer, "ln_f"):
                norm = model.transformer.ln_f
            else:
                raise ValueError(f"Incorrect Model")
            for i, r in enumerate(outputs[1].hidden_states):
                layer_logits = []
                layer_output = norm(r)
                logits = lm_head(layer_output)
                next_token_logits = logits[:, -1, :]
                top_logits_k = k_indices
                top_values, top_indices = torch.topk(next_token_logits, top_logits_k, dim=-1)
                decoded_texts = [tokenizer.decode([idx], skip_special_tokens=False) for idx in top_indices.squeeze().tolist()]
                top_values = top_values.detach().cpu()
                if decoding:
                    for value, token in zip(top_values.squeeze().tolist(), decoded_texts):
                        layer_logits.append([token, value])
                else:
                    for value, token in zip(top_values.squeeze().tolist(), top_indices.squeeze().tolist()):
                        layer_logits.append([token, value])
                tl_pair.append(layer_logits)
        res_hidden_states = []
        for _ in outputs[1].hidden_states:
            res_hidden_states.append(_.float().detach().cpu().numpy())
        return res_hidden_states, tl_pair


    def vis_heatmap(self, dataset, layer_left=0, layer_right=33, output_dir=None):
        self.forward_info = {}
        self.get_forward_info(dataset)
        topk_intermediate_confidence_heatmap(self.forward_info, topk=5, layer_nums=self.layer_nums, left=layer_left, right=layer_right, dataset_size=100, output_dir=output_dir)

            