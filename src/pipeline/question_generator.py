# src/pipeline/generate_qa.py
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
from utils.struct import DialogueTurn, Session, ConversationDataset

class QuestionGenerator:
    def __init__(self,model: str,
                 min_sessions=1, max_sessions=4,min_evidences=1, max_evidences=3):
        self.model = model
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.min_evidences = min_evidences
        self.max_evidences = max_evidences
        self.logger = logging.getLogger(__name__)
    
    # def _select_template(self, sample_type: str) -> str:
    #     """选择正确的提示模板"""
    #     if sample_type == "table":
    #         return QA_GENERATION_PROMPTS["structured_template"]
    #     return QA_GENERATION_PROMPTS["unstructured_template"]
    
    def batch_generate(self, dataset: ConversationDataset, num_qa=10) -> List[Dict]:
        """为数据集生成多个QA对"""
        all_qa = []
        for qa_index in range(num_qa):
            session_count = random.randint(self.min_sessions, min(self.max_sessions, len(dataset.sessions)))
            selected_sessions = random.sample(dataset.sessions, session_count)
            # 构建会话上下文
            session_context = self._build_session_context(selected_sessions)
            # 生成问题
            qa_response = self.generate_qa(session_context)
            if qa_response:
                # 解析响应
                qa_dict = self._parse_response(qa_response)
                
                # 添加元信息
                qa_dict["session_ids"] = [s.id for s in selected_sessions]
                qa_dict["qa_index"] = qa_index
                
                all_qa.append(qa_dict)
        return all_qa

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
            {"role": "system", "content": "You are a helpful assistant."},
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
        # 尝试解析JSON格式
        try:
            import json
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # 降级处理 - 使用正则提取关键信息
            return self._parse_fallback(response)

    def _parse_fallback(self, text: str) -> Dict:
        """应急解析方案"""
        # 使用正则表达式提取关键部分
        import re
        
        # 尝试匹配问题
        question_match = re.search(r'question[:：]?\s*(.*?)(?=\n|$)', text)
        question = question_match.group(1).strip() if question_match else "未知问题"
        
        # 尝试匹配答案
        answer_match = re.search(r'answer[:：]?\s*(.*?)(?=\n|$)', text)
        answer = answer_match.group(1).strip() if answer_match else "未知答案"
        
        # 尝试匹配证据
        evidence_match = re.findall(r'evidence[:：]?\s*(.*?)(?=\n|$)', text)
        evidence = [e.strip() for e in evidence_match] if evidence_match else ["未知证据"]
        
        return {
            "question": question,
            "answer": answer,
            "evidence": evidence
        }
def load_data(input_path: str) -> ConversationDataset:
    """加载并转换数据为ConversationDataset对象"""
    try:
        import json
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        sessions = []
        for session_data in raw_data:
            # 转换回合
            turns = []
            for turn_data in session_data.get("turns", []):
                turn = DialogueTurn(
                    turn_id=turn_data.get("turn_id", f"turn_{len(turns)+1}"),
                    speaker=turn_data.get("speaker", "Unknown"),
                    content=turn_data.get("content", "")
                )
                turns.append(turn)
            # 转换会话
            session = Session(
                session_id=session_data.get("session_id", f"session_{len(sessions)+1}"),
                time=session_data.get("time", "Unknown"),
                participants=session_data.get("participants", ["Participant A", "Participant B"]),
                turns=turns
            )
            sessions.append(session)
        return ConversationDataset(sessions=sessions)
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
        max_evidences=args.max_evidences
    )
    # 加载数据
    dataset = load_data(args.input_data)
    
    # 生成QA对
    results = qa_generator.batch_generate(dataset, num_qa=args.num_qa)
    
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