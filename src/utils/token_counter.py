import json
import tiktoken
import os
import math

class TokenCounter:
    """
    一个用于计算文本 token 数量的类。
    """
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except ValueError:
            print(f"警告: 未知编码名称 '{encoding_name}'。使用默认编码 'cl100k_base'。")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text) -> int:
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串类型。")
        return len(self.encoding.encode(text))

def print_histogram(buckets, counts, width=50):
    """
    用井号打印直方图
    buckets: 每个桶的区间字符串
    counts:  对应桶的计数
    width:   一行最多打印多少个井号
    """
    if not counts:
        return
    max_count = max(counts)
    if max_count == 0:
        max_count = 1
    for bucket, cnt in zip(buckets, counts):
        bar = "#" * int(cnt * width / max_count)
        print(f"{bucket:>10}: {bar} ({cnt})")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.input_filepath):
        print(f"错误: 输入文件 '{args.input_filepath}' 不存在。")
        return

    try:
        with open(args.input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise Exception(e)

    token_counter = TokenCounter()
    updated_data = []

    # 用于统计分布
    token_counts = []

    for conversation in data:
        full_context = []
        for session in conversation["sessions"]:
            for turn in session["turns"]:
                full_context.append(turn["content"])
        context_text = " ".join(full_context)
        token_count = token_counter.count_tokens(context_text)

        token_counts.append(token_count)
        conversation["token_count"] = token_count
        updated_data.append(conversation)
        print(f"处理 ID: {conversation['conversation_id']} - Token 数量: {token_count}")

    # 1. 计算 token 总和
    total_tokens = sum(token_counts)
    print("\n=== Token 统计 ===")
    print(f"所有对话 token 总和: {total_tokens}")
    print(f"平均对话 token 数量: {total_tokens / len(token_counts):.2f}")

    # 2. 计算 token 分布（分桶）
    if token_counts:
        min_t = min(token_counts)
        max_t = max(token_counts)
        step = 5000  # 每 100 个 token 为一桶，可自行调整
        bins = list(range((min_t // step) * step, max_t + step, step))
        buckets = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        counts = [0] * len(buckets)

        for t in token_counts:
            idx = min(t // step, len(buckets) - 1)
            counts[idx] += 1

        print("\n=== Token 分布直方图 ===")
        print_histogram(buckets, counts)

    try:
        with open(args.input_filepath, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已成功写入到 '{args.input_filepath}'。")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

if __name__ == "__main__":
    main()