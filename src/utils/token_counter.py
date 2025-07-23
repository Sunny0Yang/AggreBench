import json
import tiktoken
import os

class TokenCounter:
    """
    一个用于计算文本 token 数量的类。
    """
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        初始化 TokenCounter。
        Args:
            encoding_name: 用于计算 token 的编码名称，默认为 "cl100k_base"。
                           常用的编码包括 "cl100k_base" (GPT-3.5/GPT-4),
                           "p50k_base" (Codex, GPT-3), "r50k_base" (GPT-2, GPT-3)。
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except ValueError:
            print(f"警告: 未知编码名称 '{encoding_name}'。使用默认编码 'cl100k_base'。")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text) -> int:
        """
        计算给定文本的 token 数量。
        Args:
            text: 要计算 token 的文本字符串。
        Returns:
            文本的 token 数量。
        """
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串类型。")
        return len(self.encoding.encode(text))

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
    print("1")
    token_counter = TokenCounter()
    updated_data = []

    for conversation in data:
        full_context = []
        for session in conversation["sessions"]:
            for turn in session["turns"]:
                full_context.append(turn["content"])
        
        context_text = " ".join(full_context)
        token_count = token_counter.count_tokens(context_text)
        
        print(f"处理 ID: {conversation['conversation_id']} - Token 数量: {token_count}")
        conversation["token_count"] = token_count
        updated_data.append(conversation)

    try:
        with open(args.input_filepath, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        print(f"结果已成功写入到 '{args.input_filepath}'。")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

if __name__ == "__main__":
    main()