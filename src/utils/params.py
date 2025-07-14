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
    parser.add_argument("--generate_pseudo_dialogue", action="store_true",
                        help="是否生成伪对话")
    parser.add_argument("--model", type=str,
                        default=os.getenv('LLM_MODEL'),
                        help="LLM model name to use for session simulation.")
    parser.add_argument("--max_turns", type=int,
                        default=6,
                        help="Maximum number of turns for simulated dialogue.")
    parser.add_argument("--is_step", action="store_true",
                        help="Enable step-by-step mode for dialogue generation (pauses after each turn).")
    parser.add_argument("--cache_dir", type=str,
                        default="./dialog_cache",
                        help="Directory to cache generated dialogues.")
    parser.add_argument("--combine_size", type=int,
                        default=10,
                        help="Number of data entries to combine before processing.")
    return parser

def qa_generation_args(parser):
    parser.add_argument('input_data', type=str,
                        help='Path to dataset file')
    parser.add_argument('--model', type=str,
                        default=os.getenv('MODEL'),
                        help='LLM model to use')
    parser.add_argument('--output_dir', type=str,
                        default=os.getenv('QA_DIR'),
                        help='Output directory for QA pairs')

    parser.add_argument('--min_sessions', type=int, default=5,
                        help='Min number of sessions per conversation; related to the length of the context')
    parser.add_argument('--max_sessions', type=int, default=10,
                        help='Max number of sessions per conversation; related to the length of the context')
    parser.add_argument('--session_threshold', type=int, default=2,
                        help='Min number of sessions per QA')
    parser.add_argument('--min_evidences', type=int, default=10,
                        help='Min number of evidences per question')
    parser.add_argument('--max_evidences', type=int, default=15,
                        help='Max number of evidences per question')
    parser.add_argument("--easy", type=int, default=0,
                        help="Number of easy questions to generate.")
    parser.add_argument("--medium", type=int, default=0,
                        help="Number of medium questions to generate.")
    parser.add_argument("--hard", type=int, default=0,
                        help="Number of hard questions to generate.")

    return parser