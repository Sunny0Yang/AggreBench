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

def qa_generation_args(parser: argparse.ArgumentParser):
    """
    Defines command-line arguments for the QA generation script.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    parser.add_argument('input_data', type=str,
                        help='Path to the input dataset file (e.g., conversations.json).')
    parser.add_argument('--model', type=str,
                        default=os.getenv('MODEL', 'gemini-1.5-pro-latest'), # 提供一个默认的env变量值
                        help='Specify the LLM model to use for QA generation. Defaults to MODEL environment variable.')
    parser.add_argument('--output_dir', type=str,
                        default=os.getenv('QA_DIR', './generated_qas'), # 提供一个默认的env变量值
                        help='Output directory to store generated QA pairs and cache files. Defaults to QA_DIR environment variable.')

    # Context Selection Parameters
    parser.add_argument('--min_sessions', type=int, default=5,
                        help='Minimum number of consecutive sessions to select from a conversation for generating a QA. Relates to the context length.')
    parser.add_argument('--max_sessions', type=int, default=10,
                        help='Maximum number of consecutive sessions to select from a conversation for generating a QA. Relates to the context length.')
    parser.add_argument('--session_threshold', type=int, default=2,
                        help='Minimum number of selected sessions that must contain valid conversational turns for QA generation.')
    
    # Evidence Selection Parameters (optional, if your evidence logic is driven by this)
    parser.add_argument('--min_evidences', type=int, default=10,
                        help='Minimum number of evidence tokens/units required for a generated question. (Consider clarifying unit, e.g., "words", "sentences")')
    parser.add_argument('--max_evidences', type=int, default=15,
                        help='Maximum number of evidence tokens/units allowed for a generated question.')

    # QA Difficulty and Quantity Parameters
    parser.add_argument("--easy", type=int, default=0,
                        help="Number of 'easy' questions to generate for the entire dataset.")
    parser.add_argument("--medium", type=int, default=0,
                        help="Number of 'medium' questions to generate for the entire dataset.")
    parser.add_argument("--hard", type=int, default=0,
                        help="Number of 'hard' questions to generate for the entire dataset.")

    # Interaction and Guidance Parameters
    parser.add_argument("--is_step", action="store_true",
                        help="Enable step-by-step mode. The process will pause after each QA generation, allowing for manual review and preference setting (like/dislike).")
    parser.add_argument('--cache_dir', type=str, 
                        help='Directory to store intermediate QA cache files.')
    parser.add_argument('--max_preferred_examples', type=int, default=3,
                        help='Maximum number of "liked" QA examples to include in the LLM prompt as positive guidance. Set to 0 to exclude.')
    parser.add_argument('--max_disliked_examples', type=int, default=3,
                        help='Maximum number of "disliked" QA examples to include in the LLM prompt as negative guidance. Set to 0 to exclude.')
    parser.add_argument('--enable_validation', action='store_true', 
                        help='Enable the second stage SQL validation process.')
    parser.add_argument('--domain', type=str,
                        help='Domain of the dataset.(financial,medical)')
    # parser.add_argument('--semantic_similarity_threshold', type=float, default=0.8,
    #                     help='Cosine similarity threshold for marking a newly generated question as a semantic duplicate of an existing one. Range: 0.0 to 1.0.')
    # parser.add_argument('--embedding_model_name', type=str, default='all-MiniLM-L6-v2',
    #                     help='Name or path of the sentence-transformers model to use for semantic embedding calculations (e.g., "all-MiniLM-L6-v2").')
    return parser