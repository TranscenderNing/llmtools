import re
import ast
from tabulate import tabulate
from pathlib import Path


def merge_logs(output_file="merged.log", input_files=["log1.log", "log2.log"]):
    output_path = Path(output_file)
    with output_path.open("w") as outfile:
        for log_file in input_files:
            log_path = Path(log_file)
            outfile.write(f"\n\n=== {log_path.name} ===\n\n")
            outfile.write(log_path.read_text())


def extract_info_from_log_file(file="logfile.log"):
    # 存储提取的内容
    extracted_data = []

    with open(file, "r") as f:
        for line in f:
            # 匹配 [{'seed': 42, 'acc': ... }] 格式的行
            json_match = re.search(r"\[\{'seed': 42, 'acc':.*?\]", line)
            if json_match:
                extracted_data.append(json_match.group())

            # 匹配 "Processing model:" 开头的行
            if line.startswith("Processing model:"):
                extracted_data.append(line.strip())

    # 打印或返回提取的内容
    # for item in extracted_data:
    #     print(item)

    return extracted_data  # 可选：返回数据以便后续处理


def generate_table(data):
    # 预处理数据
    processed = []
    for layer in data:
        prune_layer = layer["prune_layer"]
        acc_dict = {}

        # 提取所有seed的acc值
        for item in layer["acc"]:
            seed = item["seed"]
            acc = list(item["acc"])[0] * 100  # 提取集合中的数值
            acc_dict[f"seed_{seed}"] = round(acc, 1)

        # 计算平均值
        values = list(acc_dict.values())
        avg = round(sum(values) / len(values), 1)

        processed.append({"prune_layer": prune_layer, **acc_dict, "平均": avg})

    print(processed)
    # 生成表格
    headers = ["prune_layer"] + sorted(
        [k for k in processed[0].keys() if k != "prune_layer" and k != "平均"],
        key=lambda x: int(x.split("_")[1]),
    )
    headers.append("平均")

    table_data = []
    for row in processed:
        table_data.append([row[col] for col in headers])

    print(
        tabulate(
            table_data,
            headers=headers,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )
    )


def main():
    # # 使用示例
    # merge_log_file = "/data/ldn/llmtools/llm-prune/logs/all_log.log"
    
    # # 将多个文件合并
    # input_files = [
    #     "/data/ldn/llmtools/llm-prune/logs/eval_cuda_1.log",
    #     "/data/ldn/llmtools/llm-prune/logs/eval_cuda_2.log",
    #     "/data/ldn/llmtools/llm-prune/logs/eval_cuda_3.log",
    #     "/data/ldn/llmtools/llm-prune/logs/eval_cuda_5.log",
    #     "/data/ldn/llmtools/llm-prune/logs/eval_cuda_6.log",
    #     "/data/ldn/llmtools/llm-prune/logs/eval_cuda_7.log",
    # ]
    # merge_logs(merge_log_file, input_files)  # 合并目录下所有.log文件


    merge_log_file = "/data/ldn/llmtools/llm-prune/logs/noprune_eval.log"
    extracted_data = extract_info_from_log_file(merge_log_file)
    print(extracted_data)

    n = len(extracted_data)
    result_arr = []
    for i in range(n):
        if i % 2 == 0:
            elem = {}
            elem["prune_layer"] = extracted_data[i].split("-")[-1]
        else:
            elem["acc"] = ast.literal_eval(extracted_data[i])
            result_arr.append(elem)
    print(result_arr)

    generate_table(result_arr)


if __name__ == "__main__":
    main()
