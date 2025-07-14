import argparse
import logging
import random
import time
import os
import re
import json
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
                 min_evidences=10, max_evidences=15):
        self.model = model
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.session_threshold = session_threshold
        self.min_evidences = min_evidences
        self.max_evidences = max_evidences
        self.logger = logging.getLogger(__name__)
        self.difficulty = "easy" # 初始化一个默认值，实际会在 batch_generate 中动态设置

    def batch_generate(self, dataset: ConversationDataset, difficulty_counts: Dict[DifficultyLevel, int]) -> List[Dict]:
        """为数据集生成多个QA对（每个对话生成固定数量QA）"""
        all_qa = []
        global_qa_idx = 0
        for conversation in dataset.conversations:
            for difficulty, num_qa_for_difficulty in difficulty_counts.items():
                if num_qa_for_difficulty == 0:
                    self.logger.debug(f" '{difficulty}' 难度的问题数量为0，跳过生成。")
                    continue
                if len(conversation.sessions) < self.min_sessions:
                    self.logger.warning(f"对话 {conversation.id} 会话数不足 ({len(conversation.sessions)} < {self.min_sessions})，跳过此对话。")
                    continue

                self.logger.info(f"开始为{conversation.id} 生成 {num_qa_for_difficulty} 个 '{difficulty}' 难度的问题...")
                self.difficulty = difficulty

                generated_count_for_current_difficulty = 0
                
                while generated_count_for_current_difficulty < num_qa_for_difficulty:
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
                    
                    qa_response = self.generate_qa(session_context)
                    
                    if qa_response:
                        qa_dict = self._parse_response(qa_response)
                        if qa_dict: # 确保解析成功
                            qa_dict["conversation_id"] = conversation.id
                            qa_dict["session_ids"] = [s.id for s in selected_sessions]
                            qa_dict["qa_index"] = global_qa_idx
                            qa_dict["participants"] = conversation.speakers
                            qa_dict["difficulty"] = self.difficulty # 添加难度信息
                            all_qa.append(qa_dict)
                            generated_count_for_current_difficulty += 1
                            global_qa_idx += 1
                            self.logger.info(f"生成了第 {generated_count_for_current_difficulty}/{num_qa_for_difficulty} 个 '{self.difficulty}' 难度QA")
                    
        return all_qa

    def _extract_session_number(self, session_id: str) -> int:
        """
        从会话ID中提取数字部分
        """
        parts = session_id.split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        self.logger.error(f"无法从session_id提取数字: {session_id}")
        return hash(session_id)

    def _build_session_context(self, sessions: List[Session]) -> str:
        """构建会话上下文表示（支持结构化数据）"""
        context = ""
        for session in sessions:
            context += f"### Session ID: {session.id}\n"
            
            if session.tables:
                self.logger.info(f"会话 {session.id} 构建表格上下文")
                context += "Data Type: Structured Table\n"
                for idx, table in enumerate(session.tables):
                    context += f"Table {idx}:\n"
                    for row_idx, row in enumerate(table.rows):
                        identifier = row.get("股票简称") or row.get("股票代码") or f"Row {row_idx}"
                        context += f"   Entity: {identifier}\n"
                        for k, v in row.items():
                            context += f"     {k}: {v}\n"
            else:
                self.logger.info(f"会话 {session.id} 构建对话上下文")
                context += f"Time: {session.time}\n"
                context += f"Participants: {', '.join(session.participants)}\n"
                context += "Dialogs:\n"
                for turn in session.turns:
                    context += f"Turn {turn.id}: {turn.speaker}: {turn.content}\n"
            
            context += "\n"
        return context

    def generate_qa(self, session_context: str) -> str:
        """生成单个QA对"""
        is_structured = "structured table" in session_context.lower()
        
        if is_structured:
            prompt_template_key = f"structured_{self.difficulty}_template_en"
            system_role = "You are a structured data analyst specializing in generating aggregation queries on structured tables."
        else:
            prompt_template_key = f"conversational_{self.difficulty}_template_en"
            system_role = "You are a conversation analyst specializing in generating aggregation queries on conversational data."
        
        if prompt_template_key not in QA_GENERATION_PROMPTS:
            self.logger.warning(f"未找到模板 '{prompt_template_key}'，使用默认模板")
            prompt_template_key = "structured_medium_template_en" if is_structured else "conversational_medium_template_en"
        
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
        self.logger.debug(f"API response: {response_content}")
        return response_content

    def _parse_response(self, response: str) -> Dict:
        """解析LLM响应为结构化字典"""
        try:
            temp = json.loads(response.strip())
            if isinstance(temp.get("answer"), str):
                match = re.search(r"^The answer is:\s*(-?[0-9]+(?:\.[0-9]+)?)", temp["answer"])
                if match:
                    temp["answer"] = float(match.group(1))
                else:
                    temp["answer"] = temp["answer"].replace("The answer is: ", "").strip()
            
            if isinstance(temp.get("evidence"), list):
                temp["evidence"] = [str(e).strip() for e in temp["evidence"]]
            elif isinstance(temp.get("evidence"), str):
                temp["evidence"] = [temp["evidence"].strip()]
            else:
                temp["evidence"] = ["未知证据"]

            return temp
        except json.JSONDecodeError:
            self.logger.error("响应解析失败")
            return None

def load_data(input_path: str) -> ConversationDataset:
    """加载并转换数据为ConversationDataset对象"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        conversations = []
        for conv_data in raw_data:
            sessions = []
            for session_data in conv_data.get("sessions", []):
                turns = []
                for turn_data in session_data.get("turns", []):
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
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"成功保存 {len(results)} 条QA对至: {output_path}")
    except Exception as e:
        raise Exception(f"保存结果到 {output_path} 时出错: {e}") from e

def main():
    logger.info(f"Starting QA generation V1 for: {args.input_data}")
    
    # 从命令行参数获取难度计数
    difficulty_counts: Dict[DifficultyLevel, int] = {
        "easy": args.easy,
        "medium": args.medium,
        "hard": args.hard
    }

    # 如果所有难度计数都为0，则设置一个默认值，例如生成10个简单问题
    if all(count == 0 for count in difficulty_counts.values()):
        logger.warning("未指定任何难度的问题数量")
        exit(0)

    # 初始化生成器
    qa_generator = QuestionGenerator(
        model=args.model,
        min_sessions=args.min_sessions,
        max_sessions=args.max_sessions,
        session_threshold=args.session_threshold,
        min_evidences=args.min_evidences,
        max_evidences=args.max_evidences,
    )
    
    # 加载数据
    dataset = load_data(args.input_data)
    logger.info(f"数据集包含 {len(dataset.conversations)} 个对话")
    
    # 生成QA对
    results = qa_generator.batch_generate(dataset, difficulty_counts)
    logger.info(f"成功生成 {len(results)} 个QA对")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    temp = (os.path.splitext(os.path.basename(args.input_data))[0]).split("_")[0]
    filename = f"{temp}_qa.json"
    output_path = os.path.join(args.output_dir, filename)
    save_results(results, output_path)
    
    logger.info(f"QA generation complete. Output to: {args.output_dir}")

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ['LOG_FILE'] = f"qa_gen_{timestamp}.log"
    logger = setup_logging()
    
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser = qa_generation_args(parser)
    args = parser.parse_args()
    
    main()