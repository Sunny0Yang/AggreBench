# src/pipeline/question_generator.py

import argparse
import logging
import random
import time
import os
from typing import List, Dict
from utils.params import get_base_parser, qa_generation_args
from utils.logger import setup_logging
from utils.prompt_templates import QA_GENERATION_PROMPTS
from client.llm_client import client
from utils.struct import MultiModalTurn, Session, Conversation, ConversationDataset

class QuestionGenerator:
    def __init__(self, model: str,
                 min_sessions=1, max_sessions=4,
                 min_evidences=1, max_evidences=3,
                 num_qa=3):
        self.model = model
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.min_evidences = min_evidences
        self.max_evidences = max_evidences
        self.num_qa = num_qa
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
            ##DEBUG
            if idx > 2:
                break
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
        """构建会话上下文表示"""
        context = ""
        for session in sessions:
            context += f"### session {session.id}\n"
            context += f"time: {session.time}\n"
            context += f"participants: {', '.join(session.participants)}\n"
            context += "dialogs:\n"
            for turn in session.turns:
                context += f"Turn_ID:{turn.id} \n {turn.speaker}: {turn.content}\n"
            context += "\n"
        return context

    def generate_qa(self, session_context: str) -> str:
        """生成单个QA对"""
        prompt = QA_GENERATION_PROMPTS["cross_session_template_en"].format(
            session_context=session_context,
            session_threshold=self.min_sessions,
            min_evidences=self.min_evidences,
            max_evidences=self.max_evidences
        )
        messages = [
            {"role": "system", "content": "You are a data analysis assistant specializing in generating queries that require aggregating information from multiple sessions."},
            {"role": "user", "content": prompt},
        ]
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

        print(f"API response: {response_content}")  # 调试用
        return response_content

    def _parse_response(self, response: str) -> Dict:
        """解析LLM响应为结构化字典"""
        try:
            import json
            import re
            temp = json.loads(response.strip())
            match = re.search(r"^The answer is: (\d+)$")
            if match:
                temp["answer"] = match.group(1)
            return temp
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            return self._parse_fallback(response)

    def _parse_fallback(self, text: str) -> Dict:
        """应急解析方案"""
        import re
        return {
            "question": re.search(r'question[:：]?\s*(.*?)(?=\n|$)', text).group(1).strip() if re.search(r'question[:：]?\s*(.*?)(?=\n|$)', text) else "未知问题",
            "answer": re.search(r'answer[:：]?\s*(.*?)(?=\n|$)', text).group(1).strip() if re.search(r'answer[:：]?\s*(.*?)(?=\n|$)', text) else "未知答案",
            "evidence": [e.strip() for e in re.findall(r'evidence[:：]?\s*(.*?)(?=\n|$)', text)] if re.findall(r'evidence[:：]?\s*(.*?)(?=\n|$)', text) else ["未知证据"]
        }

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
                
                session = Session(
                    session_id=session_data.get("session_id", f"session_{len(sessions)+1}"),
                    time=session_data.get("time", "Unknown"),
                    participants=session_data.get("participants", ["Participant A", "Participant B"]),
                    turns=turns
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
    logger.info(f"Starting QA generation for: {args.input_data}")
    
    # 初始化生成器
    qa_generator = QuestionGenerator(
        model=args.model,
        min_sessions=args.min_sessions,
        max_sessions=args.max_sessions,
        min_evidences=args.min_evidences,
        max_evidences=args.max_evidences,
        num_qa=args.num_qa
    )
    
    # 加载数据
    dataset = load_data(args.input_data)
    logger.info(f"数据集包含 {len(dataset.conversations)} 个对话")
    
    # 生成QA对
    results = qa_generator.batch_generate(dataset)
    logger.info(f"成功生成 {len(results)} 个QA对")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{os.path.splitext(os.path.basename(args.input_data))[0]}_qa.json"
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