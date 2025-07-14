import argparse
import logging
import random
import time
import os
import re
import json
from typing import List, Dict, Tuple, Any, Union
from utils.sql_engine import SqlEngine
from utils.validator import Validator
from utils.params import get_base_parser, qa_generation_args
from utils.logger import setup_logging
from utils.prompt_templates import QA_GENERATION_PROMPTS
from client.llm_client import client
from utils.struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset

# 导入v1版本的基类
from .question_generator_v1 import QuestionGenerator as QuestionGeneratorV1
from .question_generator_v1 import load_data, save_results, DifficultyLevel

class QuestionGenerator(QuestionGeneratorV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sql_engine = SqlEngine()
        self.validator = Validator()
        self.logger = logging.getLogger(__name__)

    def batch_generate(self, dataset: ConversationDataset, difficulty_counts: Dict[DifficultyLevel, int]) -> List[Dict]:
        """双阶段QA生成与验证流程"""
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
                    
                    # 第一阶段：使用LLM生成初始QA对
                    session_context = self._build_session_context(selected_sessions)
                    
                    qa_response = self.generate_qa(session_context)
                    if not qa_response:
                        self.logger.warning(f"为对话 {conversation.id} 生成QA {global_qa_idx}失败")
                        continue
                    
                    qa_dict = self._parse_response(qa_response)

                    question = qa_dict.get("question")
                    answer_llm = qa_dict.get("answer")
                    evidence_llm = qa_dict.get("evidence")
                    
                    self.logger.info(f"初始QA生成成功 - 问题: {question[:50]}...") # 打印部分问题以避免过长
                    self.logger.debug(f"Question: {question}")
                    self.logger.debug(f"Answer LLM: {answer_llm}")
                    self.logger.debug(f"Evidence LLM: {evidence_llm}")

                    # 第二阶段：SQL验证与智能修正
                    qa_dict = self.validate_and_correct(
                        question=question,
                        answer_llm=answer_llm,
                        evidence_llm=evidence_llm,
                        sessions=selected_sessions
                    )
                    
                    # 添加元信息
                    qa_dict["conversation_id"] = conversation.id
                    qa_dict["session_ids"] = [s.id for s in selected_sessions]
                    qa_dict["qa_index"] = global_qa_idx
                    qa_dict["participants"] = conversation.speakers
                    qa_dict["difficulty"] = self.difficulty

                    all_qa.append(qa_dict)
                    generated_count_for_current_difficulty += 1
                    global_qa_idx += 1
                    self.logger.info(f"成功生成并处理了第 {generated_count_for_current_difficulty}/{num_qa_for_difficulty} 个 '{self.difficulty}' 难度QA ")
        return all_qa

    def validate_and_correct(self, question: str, answer_llm: Any, 
                             evidence_llm: List[str], sessions: List[Session]) -> Dict:
        """双阶段验证"""
        result = {
            "question": question,
            "answer": answer_llm,
            "evidence": evidence_llm,
            "status": ""
        }
        
        # 创建SQL引擎
        tables = []
        for session in sessions:
            if session.tables:
                tables.extend(session.tables)
        
        if not tables:
            self.logger.warning("没有可用的表格数据")
            result["status"] = "sql_skipped"
            return result
        
        self.logger.info(f"初始化SQL引擎")
        sql_engine = SqlEngine()
        self.logger.info(f"创建表")
        sql_engine.create_table_from_struct(tables)
        
        try:
            # 生成SQL查询
            sql_prompt = self._generate_sql_prompt(question, tables)
            full_sql = self.generate_sql(sql_prompt)
            
            # 解析双查询
            answer_query, evidence_query = self.parse_double_query(full_sql)
            result["sql_answer_query"] = answer_query
            result["sql_evidence_query"] = evidence_query
            self.logger.info(f"Answer SQL: {answer_query}")
            self.logger.info(f"Evidence SQL: {evidence_query}")

            # 执行答案查询
            answer_results = sql_engine.execute_query(answer_query)
            answer_sql = answer_results[0][list(answer_results[0].keys())[0]] if answer_results else None
            self.logger.info(f"Answer SQL Result: {answer_sql}")
            # 执行证据查询
            evidence_results= sql_engine.execute_query(evidence_query)
            self.logger.info(f"Evidence SQL Result: {evidence_results}")
            # 双阶段验证
            self.logger.info(f"开始验证")
            answer_match = self.validator.compare_answers(answer_llm, answer_sql)
            evidence_match = self.validator.compare_evidence(evidence_llm, evidence_results)
            
            if answer_match and evidence_match:
                result["llm_answer"] = answer_llm
                result["llm_evidence"] = evidence_llm
                result["status"] = "match"
                return result
            if not evidence_match:
                result["status"] += "evidence not match; "
            if not answer_match:
                result["status"] += "answer not match;"
            
            result["sql_answer"] = answer_sql
            result["sql_evidence"] = evidence_results
            return result
                
        except Exception as e:
            self.logger.error(f"SQL验证失败: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    def _generate_sql_prompt(self, question: str, tables: List[Table]) -> str:
        """生成用于SQL查询的提示词"""
        # 构建表格描述
        context = ""
        for i, table in enumerate(tables):
            context += f"Table_{i} (Columns: {', '.join(table.headers)}):\n"
            if table.rows:
                row_content = (',').join(f"{k}: {v}" for k, v in table.rows[0].items())
                context += f"Sample Row: {row_content}\n"
        
        return QA_GENERATION_PROMPTS["sql_prompt_template"].format(
            question=question,
            tables=context
        )

    def generate_sql(self, prompt: str) -> str:
        """使用LLM生成双查询SQL语句"""
        messages = [
            {"role": "system", "content": "你是一个SQL专家，专门将自然语言问题转换为SQL查询。请严格按照要求返回两个SQL查询语句。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                extra_body={"enable_thinking": True}
            )
            
            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            self.logger.debug(f"SQL Generation Raw Response: \n{response}")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"SQL生成失败: {e}")
            return ""
    
    def _clean_sql(self, sql_string: str) -> str:
        """清理LLM返回的SQL字符串，移除Markdown代码块标记和多余的空白。"""
        cleaned = re.sub(r'^\s*```(?:sql)?\s*|\s*```\s*$', '', sql_string, flags=re.IGNORECASE | re.DOTALL)
        cleaned = cleaned.strip()
        statements = [s.strip() for s in cleaned.split(';') if s.strip()]
        if len(statements) > 1:
            self.logger.warning(f"检测到多个SQL语句，只使用第一个: '{cleaned}'")
            return statements[0]
        elif statements:
            return statements[0]
        else:
            return ""

    def parse_double_query(self, sql_text: str) -> Tuple[str, str]:
        """解析LLM生成的包含SQL_ANSWER和SQL_EVIDENCE的双查询字符串。"""
        answer_sql = ""
        evidence_sql = ""
        answer_match = re.search(
            r'SQL_ANSWER:\s*(.*?)(?=(?:SQL_EVIDENCE:|$))', 
            sql_text,
            re.DOTALL | re.IGNORECASE
        )

        evidence_match = re.search(
            r'SQL_EVIDENCE:\s*(.*)',
            sql_text,
            re.DOTALL | re.IGNORECASE
        )

        if answer_match:
            answer_sql = self._clean_sql(answer_match.group(1))
        if evidence_match:
            evidence_sql = self._clean_sql(evidence_match.group(1))

        if not answer_sql or not evidence_sql:
            self.logger.error(f"未能解析LLM的SQL响应，格式不正确或缺少SQL。原始文本:\n{sql_text}")
            raise ValueError("未能解析LLM的SQL响应，格式不正确或缺少SQL。")
            
        return answer_sql, evidence_sql

def main():
    logger.info(f"Starting QA generation V2 for: {args.input_data}")
    
    # Get difficulty counts from command line arguments
    difficulty_counts: Dict[DifficultyLevel, int] = {
        "easy": args.easy,
        "medium": args.medium,
        "hard": args.hard
    }

    # If no difficulty count is specified, warn and exit
    if all(count == 0 for count in difficulty_counts.values()):
        logger.warning("No difficulty level specified. Please use --easy, --medium, or --hard to specify the number of questions.")
        exit(0)

    # Initialize the QuestionGenerator (the subclass)
    qa_generator = QuestionGenerator(
        model=args.model,
        min_sessions=args.min_sessions,
        max_sessions=args.max_sessions,
        session_threshold=args.session_threshold,
        min_evidences=args.min_evidences,
        max_evidences=args.max_evidences,
    )
    
    dataset = load_data(args.input_data)
    logger.info(f"Dataset contains {len(dataset.conversations)} conversations")
    
    results = qa_generator.batch_generate(dataset, difficulty_counts)
    logger.info(f"Successfully generated {len(results)} QA pairs")
    
    os.makedirs(args.output_dir, exist_ok=True)
    temp = (os.path.splitext(os.path.basename(args.input_data))[0]).split("_")[0]
    filename = f"{temp}_qa_v2.json"
    output_path = os.path.join(args.output_dir, filename)
    save_results(results, output_path)
    
    logger.info(f"QA generation complete. Output to: {args.output_dir}")

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ['LOG_FILE'] = f"qa_gen_v2_{timestamp}.log" # Log file name for validated QA
    logger = setup_logging()
    
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser = qa_generation_args(parser)
    args = parser.parse_args()
    
    main()