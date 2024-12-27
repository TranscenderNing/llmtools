# train_gpt2_e2e.py

import os
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_metric
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. 数据准备：加载 e2e 数据集
def load_gpt2_tokenizer(model_path='/home/ldn/models/gpt2-medium'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    text = "Can you introduce bupt?"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    output = model.generate(**encoded_input)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return tokenizer, model

# 2. 数据预处理
def preprocess_function(examples, tokenizer):
    inputs = examples['input']
    targets = examples['output']
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def preprocess_data(dataset, tokenizer):
    print("开始数据预处理...")
    encoded_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    return encoded_dataset

# 3. 配置训练和模型
def setup_model_and_training():
    print("加载 GPT-2 模型和分词器...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='./results',           # 输出目录
        num_train_epochs=3,               # 训练轮次
        per_device_train_batch_size=4,    # 每设备训练batch大小
        per_device_eval_batch_size=4,     # 每设备评估batch大小
        warmup_steps=500,                 # 预热步数
        weight_decay=0.01,                # 权重衰减
        logging_dir='./logs',             # 日志目录
        logging_steps=10,
        evaluation_strategy="epoch",      # 每个epoch评估一次
    )

    return model, tokenizer, training_args

# 4. 模型训练
def train_model(model, tokenizer, train_dataset, val_dataset, training_args):
    print("开始训练模型...")
    trainer = Trainer(
        model=model,                        
        args=training_args,                 
        train_dataset=train_dataset,        
        eval_dataset=val_dataset,           
    )

    # 开始训练
    trainer.train()
    return trainer

# 5. 模型评估
def evaluate_model(trainer, test_dataset):
    print("开始评估模型...")
    results = trainer.evaluate(test_dataset)
    return results

# 6. 生成预测并解码
def generate_predictions(trainer, test_dataset, tokenizer):
    print("生成测试集预测结果...")
    predictions = trainer.predict(test_dataset)

    # 获取生成的文本
    generated_text = predictions.predictions

    # 解码生成的文本
    decoded_text = tokenizer.batch_decode(generated_text, skip_special_tokens=True)
    
    return decoded_text

# 7. 计算 ROUGE 分数
def compute_rouge(decoded_text, test_dataset):
    print("计算 ROUGE 分数...")
    rouge = load_metric('rouge')

    # 计算 ROUGE 分数
    results = rouge.compute(predictions=decoded_text, references=[example['output'] for example in test_dataset])

    return results

# 8. 主函数：执行整个训练和评估流程
def main():
    # 加载数据集
    load_gpt2_tokenizer()
    
    
    
# 9. 执行主函数
if __name__ == '__main__':
    main()
