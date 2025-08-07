import os
import re
import json
import time
import random
import logging
import argparse
from typing import Any, List, Dict, Literal, Tuple, Union

from utils.params import get_base_parser, qa_generation_args
from utils.logger import setup_logging
from utils.prompt_templates import QA_GENERATION_PROMPTS, SYSTEM_PROMPTS
from client.llm_client import client
from utils.data_struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset, load_data, save_results
from utils.cache_manager import QACacheManager, DifficultyLevel
from utils.sql_engine import SqlEngine
from utils.validator import Validator

class QuestionGenerator:
    def __init__(self, model: str,
                 min_sessions=5, max_sessions=10,
                 session_threshold=2,
                 min_evidences=10, max_evidences=15,
                 cache_dir: str = "./qa_generation_cache", is_step=False,
                 max_preferred_examples: int = 3,
                 max_disliked_examples: int = 3,
                 domain: str = "medical"):
        self.model = model
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.session_threshold = session_threshold
        self.min_evidences = min_evidences
        self.max_evidences = max_evidences
        self.logger = logging.getLogger(self.__class__.__name__)
        self.difficulty: DifficultyLevel = "easy"
        self.cache_manager = QACacheManager(cache_dir) # init includes loading cache
        self.is_step = is_step
        self.max_preferred_examples = max_preferred_examples
        self.max_disliked_examples = max_disliked_examples
        self.domain = domain  # 添加领域标识

    def batch_generate(self, dataset: ConversationDataset, difficulty_counts: Dict[DifficultyLevel, int]):
        """
        为数据集生成多个QA对（每个对话生成固定数量QA）。
        所有生成的QA都将直接管理在QACacheManager中。
        """
        if self.is_step:
            print("\n--- Step-by-step mode: QA Generation Preview. ---")
            print(f"\n您可以检查缓存文件 {self.cache_manager.current_cache_path} ，然后按回车键继续...")
            input("Press Enter to continue...")
            self.cache_manager.load_cache() # Reload cache to reflect external changes if any
        
        self.logger.info(f"Loaded {len(self.cache_manager.get_all_qas())} QAs from cache. Starting global QA index from {len(self.cache_manager.get_exportable_qas())}.")

        for conversation in dataset.conversations:
            for difficulty, num_qa_for_difficulty in difficulty_counts.items():
                if num_qa_for_difficulty == 0:
                    self.logger.debug(f"Skipping generation for '{difficulty}' difficulty as count is 0.")
                    continue

                self.logger.info(f"开始为对话 '{conversation.id}' 生成 {num_qa_for_difficulty} 个 '{difficulty}' 难度的问题...")
                self.difficulty = difficulty

                # 从cache恢复已生成的个数 F(conversation,difficulty,status)
                generated_count_for_current_difficulty = len(
                    [qa for qa in self.cache_manager.get_all_qas(difficulty=difficulty) 
                     if qa.get("conversation_id") == conversation.id and qa.get("status") in ["liked", "generated"]]
                )

                if generated_count_for_current_difficulty >= num_qa_for_difficulty:
                    self.logger.info(f"Skipping generation for '{difficulty}' difficulty as already generated {generated_count_for_current_difficulty}/{num_qa_for_difficulty} QAs.")
                    continue

                while generated_count_for_current_difficulty < num_qa_for_difficulty:
                    qa_dict, _ = self._generate_single_qa(conversation)
                    if qa_dict:
                        generated_count_for_current_difficulty += 1
                        self.logger.info(f"成功生成了第 {generated_count_for_current_difficulty}/{num_qa_for_difficulty} 个 '{self.difficulty}' 难度QA")
                    else:
                        self.logger.info(f"重新生成第{generated_count_for_current_difficulty}/{num_qa_for_difficulty} 个 '{self.difficulty}' 难度QA")

    def _generate_single_qa(self, conversation: Conversation) -> Tuple[Dict | None, List[Session]]:
        """
        生成单个QA对，并处理缓存逻辑和用户交互。
        返回生成的QA字典和所选会话列表，如果生成失败或用户拒绝则返回 (None, None)。
        """
        # 从conversation中随机选择会话
        session_count = random.randint(
            self.min_sessions, min(self.max_sessions, len(conversation.sessions))
        )
        selected_sessions = random.sample(conversation.sessions, session_count)
        selected_sessions.sort(key=lambda s: s.id)

        # Prepare context for LLM
        session_context = self._build_session_context(selected_sessions)
        self.logger.debug(f"session_context:\n{session_context}")
        # Get guidance QAs
        # positive examples (status="liked")
        preferred_qas = self.cache_manager.get_preferred_qas(self.difficulty)
        # negative examples (status="disliked")
        disliked_qas = self.cache_manager.get_disliked_qas(self.difficulty)
        additional_guidance = self._build_additional_guidance(
            preferred_qas=preferred_qas,
            disliked_qas=disliked_qas
        )
        self.logger.debug(f"additional_guidance:\n{additional_guidance}")
        qa_response = self._generate_llm_qa(session_context, additional_guidance)
        if not qa_response:
            self.logger.warning("LLM did not return a valid response.")
            return None, None

        qa_dict = self._parse_llm_response(qa_response)
        if not qa_dict:
            self.logger.warning("Failed to parse LLM response into a QA dictionary.")
            return None, None

        # Add conversation-specific and global metadata
        qa_dict["conversation_id"] = conversation.id
        qa_dict["session_ids"] = [s.id for s in selected_sessions]
        qa_dict["difficulty"] = self.difficulty
        qa_dict["qa_id"] = self.cache_manager.generate_qa_id(qa_dict)
        qa_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        qa_dict["domain"] = self.domain  # 添加领域标识

        if self.is_step:
            print("\n--- Step-by-step mode: New QA Generated. ---")
            print(f"Question: {qa_dict.get('question_text')}")
            print(f"\n请检查这次生成的问题。输入 'y' 标记为【偏好问题】（保存进数据集）；输入 'n' 标记为【不喜欢】（保存进cache），并重新生成；输入 'r' 重新生成（不保存进cache）；按回车键标记为【已生成】（保存进数据集）。")
            char = input("输入 'y', 'n', 'r'或回车键继续...\n").strip().lower()
            if char == "y":
                self.cache_manager.add_qa(qa_dict, status="liked")
                self.cache_manager.save_cache()
                self.logger.info(f"QA {qa_dict['qa_id']} marked as 'liked'.")
                return qa_dict, selected_sessions
            elif char == "n":
                self.cache_manager.add_qa(qa_dict, status="disliked")
                self.cache_manager.save_cache()
                self.logger.info(f"QA {qa_dict['qa_id']} marked as 'disliked'. Re-generating...")
                return None, None # Signal to re-generate
            elif char == "r":
                self.logger.info(f"QA {qa_dict['qa_id']} marked as 'rejected'. Re-generating...")
                return None, None # Signal to re-generate
            else:
                self.cache_manager.add_qa(qa_dict, status="generated")
                self.cache_manager.save_cache()
                self.logger.info(f"QA {qa_dict['qa_id']} marked as 'generated'.")
                return qa_dict, selected_sessions
        else:
            self.cache_manager.add_qa(qa_dict, status="generated")
            self.cache_manager.save_cache()
            self.logger.info(f"QA {qa_dict['qa_id']} added as 'generated'.")
            return qa_dict, selected_sessions

    def _build_session_context(self, sessions: List[Session]) -> str:
        context = ""
        for session in sessions:
            context += f"### Session ID: {session.id}\n"
            if session.tables:
                self.logger.debug(f"会话 {session.id} 构建表格上下文")
                context += "Data Type: Structured Table\n"
                for idx, table in enumerate(session.tables):
                    context += f"Table {idx} (Headers: {', '.join(table.headers)}):\n"
                    for row_idx, row in enumerate(table.rows):
                        # 通用行表示方法，不依赖特定列名
                        if isinstance(row, dict):
                            # 字典格式的行
                            row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
                        elif isinstance(row, list):
                            # 列表格式的行，与表头对应
                            row_str = ", ".join(f"{header}: {value}" for header, value in zip(table.headers, row))
                        else:
                            row_str = str(row)
                        
                        context += f"  Row {row_idx}: {row_str}\n"
            else:
                self.logger.debug(f"会话 {session.id} 构建对话上下文")
                context += f"Time: {session.time}\n"
                context += f"Participants: {', '.join(session.participants)}\n"
                context += "Dialogs:\n"
                for turn in session.turns:
                    context += f"Turn {turn.id}: {turn.speaker}: {turn.content}\n"
            context += "\n"
        return context

    def _build_additional_guidance(self, preferred_qas: List[Dict], disliked_qas: List[Dict]) -> str:
        """
        构建额外的指导信息，包含偏好问题、不偏好问题。
        """
        guidance = ""
        # Section 1: Crucial Instructions (Prioritize Uniqueness and Diversity)
        guidance += (
            "### IMPORTANT GENERATION GUIDELINES:\n"
            "Your primary goal is to generate a **NEW, UNIQUE, and SEMANTICALLY DISTINCT question** based on the provided context.\n"
            "**DO NOT** merely rephrase or slightly alter any existing question. Strive for **stylistic and semantic diversity**.\n"
            "Explore different facts, aspects, or aggregation types within the context to formulate truly novel questions.\n"
            "The question must be answerable solely from the provided context.\n\n"
        )

        # Section 2: Preferred Questions (Positive Examples)
        selected_preferred_qas = random.sample(preferred_qas, min(len(preferred_qas), self.max_preferred_examples))
        if selected_preferred_qas:
            guidance += (
                "### Preferred Questions (High-quality examples):\n"
                "These examples showcase the **desired characteristics** for new questions. "
                "**Crucially, DO NOT replicate their exact phrasing or merely substitute entities.** "
                "Instead, analyze them to understand the *underlying principles* of what makes them good:\n"
                "- **Question Type:** Is it a comparison, aggregation, trend analysis, specific fact retrieval?\n"
                "- **Logical Structure:** How does it connect different pieces of information?\n"
                "- **Context Utilization:** Which specific facts or data points are combined or inferred?\n"
                "- **Complexity:** How does it achieve the desired difficulty level?\n"
                "Your task is to generate **semantically unique questions** that adhere to these principles, but are distinct in their wording and specific focus.\n"
            )
            for idx, qa in enumerate(selected_preferred_qas):
                guidance += f" Good Example {idx + 1}:\n"
                guidance += f" Question: {qa.get('question_text')}\n"
                guidance += f" Answer: {qa.get('answer_text')}\n"
                guidance += "\n"

        # Section 3: Disliked Questions (Negative Examples)
        selected_disliked_qas = random.sample(disliked_qas, min(len(disliked_qas), self.max_disliked_examples))
        if selected_disliked_qas:
            guidance += (
                "### Disliked Questions (Examples to AVOID generating):\n"
                "These examples were deemed low-quality, irrelevant, or undesirable. "
                "**Pay close attention to what makes them bad.** Is it due to ambiguity, lack of answerability, redundancy, or irrelevant details?\n"
                "**Absolutely DO NOT generate questions that are semantically identical or very similar to these in form or content.** "
                "Learn from their flaws to prevent similar mistakes in your new questions.\n"
            )
            for idx, qa in enumerate(selected_disliked_qas):
                guidance += f" Bad Example {idx + 1}:\n"
                guidance += f" Question: {qa.get('question_text', 'N/A')}\n"
                guidance += "\n"
        return guidance

    def _generate_llm_qa(self, session_context: str, additional_guidance: str) -> str:
        """生成单个QA对"""
        domain = self.domain
        # 获取系统提示
        system_role = SYSTEM_PROMPTS.get(domain, "")
        template_key = f"{domain}_structured_{self.difficulty}_template_en"

        if template_key not in QA_GENERATION_PROMPTS:
            self.logger.error(f"未找到模板 '{template_key}'，使用默认模板")
            raise ValueError(f"未找到模板 '{template_key}'")
        prompt = QA_GENERATION_PROMPTS[template_key].format(
            session_context=session_context,
            session_threshold=self.session_threshold,
            min_evidences=self.min_evidences,
            max_evidences=self.max_evidences
        )
        if additional_guidance:
            prompt += additional_guidance

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]
        self.logger.debug(f"Prompt:{messages}")
        self.logger.info(f"正在为难度 '{self.difficulty}' 生成QA...")
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            extra_body={"enable_thinking": True}
        )

        response_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
        self.logger.debug(f"API response: {response_content}")
        return response_content

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的 JSON 字符串，返回结构化字典：
        {
            "question_text": str,
            "answer_text": float | int,
            "evidence": List[Dict]  # 改为字典列表，更灵活
        }
        """
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # 移除开头的```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # 移除结尾的```
            cleaned_response = cleaned_response.strip()

            data = json.loads(cleaned_response)

            question_text = data.pop("question", None)
            answer_text   = data.pop("answer", None)

            if isinstance(answer_text, str):
                num_match = re.search(r"-?\d+(?:\.\d+)?", answer_text)
                if num_match:
                    answer_text = float(num_match.group(0))
                else:
                    answer_text = 0.0
            elif isinstance(answer_text, (int, float)):
                answer_text = float(answer_text)
            else:
                answer_text = 0.0

            # 使用字典列表存储证据，更灵活
            evidence = data.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            
            # 确保每个证据项是字典
            validated_evidence = []
            for item in evidence:
                if isinstance(item, dict):
                    validated_evidence.append(item)
                elif isinstance(item, list):
                    # 尝试转换为字典
                    try:
                        if len(item) == 2:  # 键值对列表
                            validated_evidence.append(dict(item))
                        else:  # 无法转换，保留原始格式
                            validated_evidence.append({"raw": item})
                    except:
                        validated_evidence.append({"raw": item})
                else:
                    validated_evidence.append({"raw": item})
            
            return {
                "question_text": question_text,
                "answer_text": answer_text,
                "evidence": validated_evidence
            }
        except Exception as e:
            self.logger.error(f"解析响应时发生错误: {e}")
            return None

class BatchValidator:
    """
    负责对已生成的问题答案对进行SQL验证和智能修正的类。
    """
    def __init__(self, model: str, domain: str = "financial"):
        self.sql_engine = SqlEngine()
        self.validator = Validator()
        self.model = model
        self.domain = domain  # 添加领域标识
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_qas(self, cache_manager: QACacheManager, dataset: ConversationDataset, is_step: bool):
        """
        遍历缓存中的QA对，进行SQL验证和修正。
        """
        self.logger.info("Starting QA validation process.")
        
        if is_step:
            print("\n--- Step-by-step mode: QA Validation Preview. ---")
            print(f"\n您可以检查缓存文件 {cache_manager.current_cache_path} ，然后按回车键继续...")
            input("Press Enter to continue...")
            cache_manager.load_cache() # Reload cache before validation step
        # Get all generated/liked QAs that haven't been successfully SQL verified
        qas_to_validate = [
            qa for qa in cache_manager.get_exportable_qas()
            if qa.get("sql_info", {}).get("sql_status") not in {"match","evidence_not_match"}
        ]
        self.logger.info(f"Found {len(qas_to_validate)} QAs to validate.")

        for qa_item in qas_to_validate:
            question = qa_item.get("question_text")
            answer_llm = qa_item.get("answer_text")
            evidence_llm = qa_item.get("evidence", [])

            conversation_id = qa_item.get("conversation_id")
            session_ids = qa_item.get("session_ids")

            if not all([question, answer_llm, evidence_llm, conversation_id, session_ids]):
                self.logger.warning(f"Skipping malformed QA item: {qa_item.get('qa_id')}")
                sql_info = {"sql_status": "skipped", "sql_error": "malformed_qa"}
            else:
                # Find the relevant sessions from the dataset
                relevant_conversation = next((conv for conv in dataset.conversations if conv.id == conversation_id), None)
                if not relevant_conversation:
                    self.logger.warning(f"Conversation {conversation_id} not found for QA {qa_item.get('qa_id')}. Skipping validation.")
                    cache_manager.add_qa(qa_item, status=qa_item.get("status"), sql_info={"sql_status": "skipped", "sql_error": "no conversation"})
                    cache_manager.save_cache()
                    continue

                selected_sessions = [s for s in relevant_conversation.sessions if s.id in session_ids]
                if not selected_sessions:
                    self.logger.warning(f"Sessions {session_ids} not found for QA {qa_item.get('qa_id')}. Skipping validation.")
                    cache_manager.add_qa(qa_item, status=qa_item.get("status"), sql_info={"sql_status": "skipped", "sql_error": "no sessions"})
                    cache_manager.save_cache()
                    continue
                
                self.logger.info(f"Validating QA: {qa_item.get('qa_id')}")

                # Perform validation and correction
                sql_info = self.validate_and_correct(
                                        question=question,
                                        answer_llm=answer_llm,
                                        evidence_llm=evidence_llm,
                                        sessions=selected_sessions
                                    )
            
            # Update cache with validation results
            cache_manager.add_qa(qa_item, status=qa_item.get("status"), sql_info = sql_info)
            cache_manager.save_cache()
            self.logger.info(f"QA {qa_item.get('qa_id')} validation complete. Status: {sql_info.get('sql_status')}")

    def validate_and_correct(self, question: str, answer_llm: Any, 
                             evidence_llm: List[Dict], sessions: List[Session]) -> Dict:
        """
        执行验证：LLM生成的SQL查询与数据库结果对比。
        """
        result = {
            "sql_status": "not_yet",
            "sql_answer_query": None,
            "sql_evidence_query": None,
            "sql_answer": None,
            "sql_evidence": None,
            "error": None
        }
        
        tables = []
        for session in sessions:
            if session.tables:
                tables.extend(session.tables)
        if not tables:
            self.logger.warning(f"No tables available, skipping SQL validation.")
            result["sql_status"] = "skipped"
            return result
        
        try:
            # 创建统一表
            self.logger.debug(f"###Tables###:{tables}")
            self.sql_engine.create_table_from_struct(tables, domain=self.domain)
            
            # 生成SQL查询
            sql_prompt = self._generate_sql_prompt(question, self.domain)
            full_sql = self._generate_sql_from_llm(sql_prompt)
            
            # 解析双查询
            answer_query, evidence_query = self._parse_double_query(full_sql)
            result["sql_answer_query"] = answer_query
            result["sql_evidence_query"] = evidence_query
            
            # 执行答案查询
            answer_results = self.sql_engine.execute_query(answer_query)
            answer_sql = answer_results[0][list(answer_results[0].keys())[0]] if answer_results and answer_results[0] else None
            result["sql_answer"] = answer_sql
            
            # 执行证据查询
            evidence_results = self.sql_engine.execute_query(evidence_query)
            result["sql_evidence"] = evidence_results
            
            # 执行验证
            answer_match = self.validator.compare_answers(answer_llm, answer_sql)
            evidence_match = self.validator.compare_evidence(evidence_llm, evidence_results, self.domain)
            
            if answer_match and evidence_match:
                result["sql_status"] = "match"
            else:
                status_parts = []
                if not answer_match:
                    status_parts.append("answer_not_match")
                if not evidence_match:
                    status_parts.append("evidence_not_match")
                result["sql_status"] = "; ".join(status_parts)
            
        except Exception as e:
            self.logger.error(f"SQL validation failed: {str(e)}")
            result["sql_status"] = "failed"
            result["error"] = str(e)
        
        return result

    def _generate_sql_prompt(self, question: str, domain: str) -> str:
        """根据领域生成SQL提示"""
        if domain == "financial":
            return f"""
            ### Financial SQL Generation
            You are a financial SQL expert. Generate two SQL queries for the following question:
            Question: {question}
            
            Database Schema:
            Table: unified_data
            Columns: code (TEXT), sname (TEXT), tdate (TEXT), value (REAL), metric (TEXT)
            
            Requirements:
            1. First query (SQL_ANSWER): Calculate the answer to the question.
            2. Second query (SQL_EVIDENCE): Retrieve all evidence rows used in the calculation.
            3. Output both queries in the format:
                SQL_ANSWER: [query];
                SQL_EVIDENCE: [query];
            """
        elif domain == "medical":
            return f"""
            ### Medical SQL Generation
            You are a medical SQL expert. Generate two SQL queries for the following question:
            Question: {question}
            
            Database Schema:
            Table: unified_data
            Columns: patient_id (TEXT), timestamp (TEXT), variable_name (TEXT), value (REAL), table_type (TEXT)
            
            Requirements:
            1. First query (SQL_ANSWER): Calculate the answer to the question.
            2. Second query (SQL_EVIDENCE): Retrieve all evidence rows used in the calculation.
            3. Output both queries in the format:
                SQL_ANSWER: [query];
                SQL_EVIDENCE: [query];
            """
        else:
            return f"""
            ### Generic SQL Generation
            Generate two SQL queries for the following question:
            Question: {question}
            
            Database Schema:
            Table: unified_data
            Columns: entity_id (TEXT), timestamp (TEXT), variable_name (TEXT), value (REAL), table_type (TEXT)
            
            Requirements:
            1. First query (SQL_ANSWER): Calculate the answer to the question.
            2. Second query (SQL_EVIDENCE): Retrieve all evidence rows used in the calculation.
            3. Output both queries in the format:
                SQL_ANSWER: [query];
                SQL_EVIDENCE: [query];
            """

    def _generate_sql_from_llm(self, prompt: str) -> str:
        """使用LLM生成双查询SQL语句"""
        messages = [
            {"role": "system", "content": "You are an SQL expert. Generate two SQL queries: one for the answer and one for the evidence."},
            {"role": "user", "content": prompt}
        ]
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                extra_body={"enable_thinking": False}
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"SQL generation failed: {e}")
            return None

    def _parse_double_query(self, full_sql: str) -> Tuple[str, str]:
        """解析LLM生成的包含SQL_ANSWER和SQL_EVIDENCE的双查询字符串"""
        # 尝试解析标准格式
        answer_match = re.search(r'SQL_ANSWER:\s*(.*?)(?=SQL_EVIDENCE:|$)', full_sql, re.DOTALL | re.IGNORECASE)
        evidence_match = re.search(r'SQL_EVIDENCE:\s*(.*)', full_sql, re.DOTALL | re.IGNORECASE)
        
        if answer_match and evidence_match:
            return self._clean_sql(answer_match.group(1)), self._clean_sql(evidence_match.group(1))
        
        # 尝试解析SQL语句
        sql_statements = [s.strip() for s in full_sql.split(';') if s.strip()]
        if len(sql_statements) >= 2:
            return sql_statements[0], sql_statements[1]
        
        # 无法解析，返回错误
        self.logger.error(f"Failed to parse SQL response: {full_sql}")
        raise ValueError("Failed to parse SQL response")

    def _clean_sql(self, sql_string: str) -> str:
        """清理SQL字符串，移除Markdown代码块标记和多余空格"""
        cleaned = re.sub(r'^\s*```(?:sql)?\s*|\s*```\s*$', '', sql_string, flags=re.IGNORECASE)
        return cleaned.strip()

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ['LOG_FILE'] = f"qa_gen_{timestamp}.log"
    logger = setup_logging()

    parser = get_base_parser()
    qa_generation_args(parser)
    args = parser.parse_args()
    
    logger.info("Starting QA generation process.")
    try:
        # Load data
        dataset = load_data(args.input_data)
        logger.info(f"数据集包含 {len(dataset.conversations)} 个对话")
        
        difficulty_counts: Dict[DifficultyLevel, int] = {
            "easy": args.easy,
            "medium": args.medium,
            "hard": args.hard
        }
        # 如果所有难度计数都为0
        if all(count == 0 for count in difficulty_counts.values()):
            logger.warning("No difficulty level specified. Please use --easy, --medium, or --hard to specify the number of questions.")
            exit(0)
        # Initialize QuestionGenerator (generation phase)
        qa_generator = QuestionGenerator(
            model=args.model,
            min_sessions=args.min_sessions,
            max_sessions=args.max_sessions,
            session_threshold=args.session_threshold,
            min_evidences=args.min_evidences,
            max_evidences=args.max_evidences,
            is_step=args.is_step,
            max_preferred_examples=args.max_preferred_examples,
            max_disliked_examples=args.max_disliked_examples,
            cache_dir=args.cache_dir,
            domain=args.domain,
        )
        
        qa_generator.batch_generate(dataset, difficulty_counts) 

        # --- Second Stage: Validation (Optional) ---
        if args.enable_validation:
            logger.info("Enabling QA validation process.")
            batch_validator = BatchValidator(model=args.model, domain=args.domain)
            batch_validator.validate_qas(qa_generator.cache_manager, dataset, args.is_step)
            logger.info("QA validation process completed.")

        # 从缓存中获取所有可导出的QA对 (liked or generated)
        results = qa_generator.cache_manager.get_exportable_qas()
        for idx, qa_item in enumerate(results):
            qa_item["qa_index"] = idx
        logger.info(f"成功从缓存中收集到 {len(results)} 个QA对准备导出")

    except Exception as e:
        logger.error(f"An error occurred during QA processing: {e}", exc_info=True)
    finally:
        if 'results' not in locals():
            results = []
        os.makedirs(args.output_dir, exist_ok=True)
        # Create a dynamic filename based on input data and validation status
        temp_filename = os.path.splitext(os.path.basename(args.input_data))[0]
        output_filename = f"{temp_filename}_qas_validated.json" if args.enable_validation else f"{temp_filename}_qas_generated.json"
        
        output_path = os.path.join(args.output_dir, output_filename)
        save_results(results, output_path)
        logger.info(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    main()