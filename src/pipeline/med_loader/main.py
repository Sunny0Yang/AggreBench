import argparse
import time
from utils.logger import setup_logging
from pipeline.med_loader.medical_preprocessor import MedicalPreprocessor
from pipeline.med_loader.medical_dialogue_generator import MedicalDialogueGenerator
import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
from utils.data_struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset
from utils.session_simulator import SessionSimulator
from utils.prompt_templates import PERSONA
def main():
    # 设置日志
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ['LOG_FILE'] = f"medical_processing_{timestamp}.log"
    logger = setup_logging()
    
    # 参数解析
    parser = argparse.ArgumentParser(description='医疗数据处理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='预处理医疗数据')
    preprocess_parser.add_argument('--input_dir', type=str, required=True,
                                   help='原始数据目录路径')
    preprocess_parser.add_argument('--output_dir', type=str, required=True,
                                   help='预处理输出目录')
    preprocess_parser.add_argument('--max_events', type=int, default=8,
                                   help='每个会话的最大事件数')
    preprocess_parser.add_argument('--time_window', type=int, default=3,
                                   help='时间窗口大小（小时）')
    
    # 对话生成命令
    generate_parser = subparsers.add_parser('generate', help='生成伪对话')
    generate_parser.add_argument('--input_dir', type=str, required=True,
                                 help='预处理数据目录路径')
    generate_parser.add_argument('--output_dir', type=str, required=True,
                                 help='最终输出目录')
    generate_parser.add_argument('--model', type=str, default='qwen-turbo-latest',
                                 help='用于对话生成的模型')
    generate_parser.add_argument('--cache_dir', type=str, default='cache/medical',
                                 help='缓存生成对话的目录')
    generate_parser.add_argument('--max_turns', type=int, default=5,
                                 help='最大对话轮数')
    generate_parser.add_argument('--is_step', action='store_true',
                                 help='启用逐步生成')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        # 运行预处理
        preprocessor = MedicalPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_events_per_session=args.max_events,
            time_window_hours=args.time_window
        )
        preprocessor.preprocess()
    
    elif args.command == 'generate':
        # 运行对话生成
        generator = MedicalDialogueGenerator(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=args.model,
            cache_dir=args.cache_dir,
            max_turns=args.max_turns,
            is_step=args.is_step
        )
        generator.generate_dialogues()
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()