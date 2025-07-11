import logging
import random
from typing import List, Dict, Tuple, Any, Union
from utils.sql_engine import SqlEngine
from utils.validator import Validator
from utils.prompt_templates import QA_GENERATION_PROMPTS
from client.llm_client import client
from utils.struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset
import re
import math

# 导入v1版本的基类
from .question_generator_v1 import QuestionGenerator as QuestionGeneratorV1
from .question_generator_v1 import load_data, save_results

class QuestionGenerator(QuestionGeneratorV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sql_engine = SqlEngine()
        self.validator = Validator()
        self.logger = logging.getLogger(__name__)

    def batch_generate(self, dataset: ConversationDataset) -> List[Dict]:
        """双阶段QA生成与验证流程"""
        all_qa = []
        idx = 0
        for conversation in dataset.conversations:
            # 跳过会话数量不足的对话
            if len(conversation.sessions) < self.min_sessions:
                self.logger.warning(f"对话 {conversation.id} 会话数不足 ({len(conversation.sessions)} < {self.min_sessions})，跳过")
                continue
            ##DEBUG
            if idx > 1:
                break

            for qa_index in range(self.num_qa):
                # 选择会话和构建上下文
                # session_count = random.randint(
                #     self.min_sessions, 
                #     min(self.max_sessions, len(conversation.sessions))
                # )
                # selected_sessions = random.sample(conversation.sessions, session_count)
                # selected_sessions.sort(key=lambda s: self._extract_session_number(s.id))
                
                selected_sessions = conversation.sessions

                # 第一阶段：使用LLM生成初始QA对
                session_context = self._build_session_context(selected_sessions)
                qa_response = self.generate_qa(session_context)
                if not qa_response:
                    self.logger.warning(f"QA{qa_index}生成失败")
                    continue
                
                qa_dict = self._parse_response(qa_response)
                question = qa_dict.get("question")
                answer_llm = qa_dict.get("answer")
                evidence_llm = qa_dict.get("evidence")
                
                self.logger.info(f"QA{qa_index}生成成功")
                self.logger.info(f"Question: {question}")
                self.logger.info(f"Answer LLM: {answer_llm}")
                self.logger.info(f"Evidence LLM: {evidence_llm}")

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
                qa_dict["qa_index"] = qa_index + idx * self.num_qa
                qa_dict["participants"] = conversation.speakers
                
                all_qa.append(qa_dict)
            
            idx += 1
        
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
                result["answer"] = answer_llm
                result["evidence"] = evidence_llm
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
            context += f"Table_{i} ({', '.join(table.headers)}):\n"
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
            self.logger.info(f"SQL Generation: {response}")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"SQL生成失败: {e}")
            return ""
    
    def _clean_sql(self, sql_string: str) -> str:
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
            self.logger.warning(f"无效的双查询格式或未能成功解析SQL。原始文本: {sql_text}")
            raise ValueError("未能解析LLM的SQL响应，格式不正确或缺少SQL。")
        return answer_sql, evidence_sql

def main():
    logger.info(f"Starting QA generation V2 for: {args.input_data}")
    
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
    import time
    import os
    import argparse
    from utils.params import get_base_parser, qa_generation_args
    from utils.logger import setup_logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ['LOG_FILE'] = f"qa_gen_{timestamp}.log"
    logger = setup_logging()
    
    # 参数解析
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser = qa_generation_args(parser)
    args = parser.parse_args()
    
    main()