# src/pipeline/question_generator.py

import argparse
import logging
import random
import time
import os
import re
from typing import Any, List, Dict, Literal
from utils.params import get_base_parser, qa_generation_args
from utils.logger import setup_logging
from utils.prompt_templates import QA_GENERATION_PROMPTS
from client.llm_client import client
from utils.struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset

DifficultyLevel = Literal["easy", "medium", "hard"]

class QuestionGenerator:
    def __init__(self, model: str,
                 min_sessions=5, max_sessions=10,
                 session_threshold=2,
                 min_evidences=10, max_evidences=15,
                 num_qa=3,
                 difficulty: DifficultyLevel = "easy"):
        self.model = model
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.session_threshold = session_threshold
        self.min_evidences = min_evidences
        self.max_evidences = max_evidences
        self.num_qa = num_qa
        self.difficulty = difficulty
        self.logger = logging.getLogger(__name__)
    
    def batch_generate(self, dataset: ConversationDataset) -> List[Dict]:
        """为数据集生成多个QA对（每个对话生成固定数量QA）"""
        all_qa = []
        idx = 0
        for conversation in dataset.conversations:
            idx += 1
            # 跳过会话数量不足的对话
            if len(conversation.sessions) < self.min_sessions:
                self.logger.warning(f"对话 {conversation.id} 会话数不足 ({len(conversation.sessions)} < {self.min_sessions})，跳过")
                continue
            # 为每个对话生成指定数量的QA对
            for qa_index in range(self.num_qa):
                # 在对话内随机选择会话
                session_count = random.randint(
                    self.min_sessions, 
                    min(self.max_sessions, len(conversation.sessions))
                )
                selected_sessions = random.sample(conversation.sessions, session_count)
                selected_sessions.sort(key=lambda s: self._extract_session_number(s.id))
                # 构建会话上下文
                # TODO
                # if self.difficulty == "hard":
                #     # 可以在这里引入额外的“无关”会话，或者从整个对话中选择更多分散的会话
                #     # 比如，除了 selected_sessions 外，再随机选择一些非重叠的 sessions 作为干扰
                #     num_irrelevant_sessions = random.randint(1, 3) # 随机加入1-3个不相关的会话作为干扰
                #     all_available_sessions = [s for s in conversation.sessions if s not in selected_sessions]
                #     irrelevant_sessions = random.sample(all_available_sessions, min(num_irrelevant_sessions, len(all_available_sessions)))
                    
                #     # 将不相关会话随机插入到选定会话中，模拟噪音
                #     combined_sessions = selected_sessions + irrelevant_sessions
                #     random.shuffle(combined_sessions) # 打乱顺序，使相关信息更难提取
                #     session_context = self._build_session_context(combined_sessions)
                # else:
                #     session_context = self._build_session_context(selected_sessions)

                session_context = self._build_session_context(selected_sessions)
                
                # 生成问题
                qa_response = self.generate_qa(session_context)
                
                if qa_response:
                    # 解析响应
                    qa_dict = self._parse_response(qa_response)
                    
                    # 添加元信息
                    qa_dict["conversation_id"] = conversation.id
                    qa_dict["session_ids"] = [s.id for s in selected_sessions]
                    qa_dict["qa_index"] = qa_index + idx * self.num_qa
                    qa_dict["participants"] = conversation.speakers
                    all_qa.append(qa_dict)
        
        return all_qa

    def _extract_session_number(self, session_id: str) -> int:
        """
        从会话ID中提取数字部分
        """
        # 处理两种可能格式：
        # 1. 简单格式: "session_6"
        # 2. 完整格式: "conv_1_session_6"
        parts = session_id.split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        logger.error(f"无法从session_id提取数字: {session_id}")
        return hash(session_id)

    def _build_session_context(self, sessions: List[Session]) -> str:
        """构建会话上下文表示（支持结构化数据）"""
        context = ""
        for session in sessions:
            context += f"### Session ID: {session.id}\n"
            
            # 检测是否是表格数据
            if session.tables:
                self.logger.info(f"会话 {session.id} 构建表格上下文")
                context += "Data Type: Structured Table\n"
                for idx, table in enumerate(session.tables):
                    context += f"Table {idx}:\n" # Keep Table ID for context if multiple tables per session
                    for row_idx, row in enumerate(table.rows):
                        # Attempt to find a '股票简称' or '股票代码' as a primary identifier for the row
                        identifier = row.get("股票简称") or row.get("股票代码") or f"Row {row_idx}"
                        context += f"  Entity: {identifier}\n" # Identify the entity for the row
                        for k, v in row.items():
                            # Present each column as "Column_Name: Value"
                            context += f"    {k}: {v}\n"
            else:
                # 常规对话处理
                self.logger.info(f"会话 {session.id} 构建对话上下文")
                context += f"Time: {session.time}\n"
                context += f"Participants: {', '.join(session.participants)}\n"
                context += "Dialogs:\n"
                for turn in session.turns:
                    context += f"Turn {turn.id}: {turn.speaker}: {turn.content}\n"
            
            context += "\n"

        # print(context)  # DEBUG
        # exit(0)
        return context

    def generate_qa(self, session_context: str) -> str:
        """生成单个QA对"""
        # 检测是否是结构化数据
        is_structured = "structured table" in session_context.lower()
        # print(f"Is structured: {is_structured}")  # DEBUG
        # 选择模板
        if is_structured:
            prompt_template_key = f"structured_{self.difficulty}_template_en"
            system_role = "You are a structured data analyst specializing in generating aggregation queries on structured tables."
        else:
            prompt_template_key = f"conversational_{self.difficulty}_template_en"
            system_role = "You are a conversation analyst specializing in generating aggregation queries on conversational data."
        
        # 模板回退逻辑
        if prompt_template_key not in QA_GENERATION_PROMPTS:
            self.logger.warning(f"未找到模板 '{prompt_template_key}'，使用默认模板")
            prompt_template_key = "structured_medium_template_en" if is_structured else "conversational_medium_template_en"
        
        # 构建提示词
        prompt = QA_GENERATION_PROMPTS[prompt_template_key].format(
            session_context=session_context,
            session_threshold=self.session_threshold,
            min_evidences=self.min_evidences,
            max_evidences=self.max_evidences
        )
        
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]
        self.logger.info(f"正在为难度 '{self.difficulty}' 生成QA...")
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            extra_body={"enable_thinking": True}  # 设置为False避免开源版报错
        )

        response_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content

        return response_content

    def _parse_response(self, response: str) -> Dict:
        """解析LLM响应为结构化字典"""
        try:
            import json
            import re
            temp = json.loads(response.strip())
            # 确保 'answer' 字段的解析能处理数字或字符串
            if isinstance(temp.get("answer"), str):
                match = re.search(r"^The answer is:\s*(-?[0-9]+(?:\.[0-9]+)?)", temp["answer"])
                if match:
                    temp["answer"] = float(match.group(1))
                else:
                    temp["answer"] = temp["answer"].replace("The answer is: ", "").strip()
            
            # 确保 'evidence' 是列表，且每个元素都是字符串
            if isinstance(temp.get("evidence"), list):
                temp["evidence"] = [str(e).strip() for e in temp["evidence"]]
            elif isinstance(temp.get("evidence"), str):
                temp["evidence"] = [temp["evidence"].strip()] # 如果是字符串，尝试转为列表
            else:
                temp["evidence"] = ["未知证据"] # 兜底

            return temp
        except json.JSONDecodeError:
            self.logger.error("解析响应失败，尝试使用应急解析方案")
            return None

def load_data(input_path: str) -> ConversationDataset:
    """加载并转换数据为ConversationDataset对象"""
    try:
        import json
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        conversations = []
        for conv_data in raw_data:
            sessions = []
            for session_data in conv_data.get("sessions", []):
                turns = []
                for turn_data in session_data.get("turns", []):
                    # 使用MultiModalTurn处理多模态信息
                    turn = MultiModalTurn(
                        turn_id=turn_data.get("turn_id", f"turn_{len(turns)+1}"),
                        speaker=turn_data.get("speaker", "Unknown"),
                        content=turn_data.get("content", "")
                    )
                    turns.append(turn)
                
                tables = []
                for table_data in session_data.get("tables", []):
                    headers = table_data.get("headers", [])
                    rows = table_data.get("rows", [])
                    table = Table(headers=headers, rows=rows)
                    tables.append(table)

                session = Session(
                    session_id=session_data.get("session_id", f"session_{len(sessions)+1}"),
                    time=session_data.get("time", "Unknown"),
                    participants=session_data.get("participants", ["Participant A", "Participant B"]),
                    turns=turns,
                    type=session_data.get("type", "conversation"),
                    tables=tables
                )
                sessions.append(session)
            
            conversation = Conversation(
                conversation_id=conv_data.get("conversation_id", f"conv_{len(conversations)+1}"),
                speakers=conv_data.get("speakers", ["Speaker A", "Speaker B"]),
                sessions=sessions
            )
            conversations.append(conversation)
        
        return ConversationDataset(conversations=conversations)
    except Exception as e:
        raise RuntimeError(f"数据加载错误: {str(e)}")
        
def save_results(results: List[Dict], output_path: str):
    """保存生成的QA对结果"""
    try:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"成功保存 {len(results)} 条QA对至: {output_path}")
    except Exception as e:
        raise Exception(f"保存结果到 {output_path} 时出错: {e}") from e

def main():
    logger.info(f"Starting QA generation V1 for: {args.input_data}")
    
    # 初始化生成器
    qa_generator = QuestionGenerator(
        model=args.model,
        min_sessions=args.min_sessions,
        max_sessions=args.max_sessions,
        session_threshold=args.session_threshold,
        min_evidences=args.min_evidences,
        max_evidences=args.max_evidences,
        num_qa=args.num_qa,
        difficulty=args.difficulty
    )
    
    # 加载数据
    dataset = load_data(args.input_data)
    logger.info(f"数据集包含 {len(dataset.conversations)} 个对话")
    
    # 生成QA对
    results = qa_generator.batch_generate(dataset)
    logger.info(f"成功生成 {len(results)} 个QA对")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    temp = (os.path.splitext(os.path.basename(args.input_data))[0]).split("_")[0]
    filename = f"{temp}_{args.difficulty}_qa.json"
    output_path = os.path.join(args.output_dir, filename)
    save_results(results, output_path)
    
    logger.info(f"QA generation complete. Output to: {args.output_dir}")

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ['LOG_FILE'] = f"qa_gen_{timestamp}.log"
    logger = setup_logging()
    
    # 参数解析
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser = qa_generation_args(parser)
    args = parser.parse_args()
    
    main()