import argparse
import os

def get_base_parser():
    """基础参数解析器"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--log_level', type=str, default=os.getenv('LOG_LEVEL', 'INFO'),
                        help='Logging level (DEBUG/INFO/WARNING/ERROR)')
    return parser

def data_loader_args(parser):
    parser.add_argument('input_data', type=str,
                        help='Path to dataset file')
    parser.add_argument('--output_dir', type=str,
                        default=os.getenv('PROCESSED_DIR'),
                        help='Output directory for processed data')
    return parser

def qa_generation_args(parser):
    parser.add_argument('input_data', type=str, 
                        help='Path to dataset file')
    parser.add_argument('--model', type=str,
                        default=os.getenv('MODEL','qwen3'),
                        help='LLM model to use')
    parser.add_argument('--output_dir', type=str, 
                        default=os.getenv('QA_DIR'),
                        help='Output directory for QA pairs')

    parser.add_argument('--min_sessions', type=int, default=1,
                        help='Min number of sessions per QA')
    parser.add_argument('--max_sessions', type=int, default=2,
                        help='Max number of sessions per QA')
    parser.add_argument('--min_evidences', type=int, default=5,
                        help='Min number of evidences per question')
    parser.add_argument('--max_evidences', type=int, default=10,
                        help='Max number of evidences per question')
    parser.add_argument('--num_qa', type=int, default=20,
                        help='Number of QA pairs to generate')
    return parser


# 类似的其他阶段的参数函数
# add_session_generation_args(), add_composition_args()...