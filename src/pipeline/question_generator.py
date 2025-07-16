import os
import re
import json
import time
import random
import logging
import argparse
from typing import Any, List, Dict, Literal, Tuple

from utils.params import get_base_parser, qa_generation_args
from utils.logger import setup_logging
from utils.prompt_templates import QA_GENERATION_PROMPTS
from client.llm_client import client
from utils.struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset, load_data, save_results
from utils.cache_manager import QACacheManager, DifficultyLevel # Import DifficultyLevel from cache_manager

class QuestionGenerator:
    def __init__(self, model: str,
                 min_sessions=5, max_sessions=10,
                 session_threshold=2,
                 min_evidences=10, max_evidences=15,
                 cache_dir: str = "./qa_generation_cache", is_step=False,
                 max_preferred_examples: int = 3,
                 max_disliked_examples: int = 3):
        self.model = model
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.session_threshold = session_threshold
        self.min_evidences = min_evidences
        self.max_evidences = max_evidences
        self.logger = logging.getLogger(__name__)
        self.difficulty: DifficultyLevel = "easy"
        self.cache_manager = QACacheManager(cache_dir)
        self.is_step = is_step
        self.max_preferred_examples = max_preferred_examples
        self.max_disliked_examples = max_disliked_examples

    def batch_generate(self, dataset: ConversationDataset, difficulty_counts: Dict[DifficultyLevel, int]):
        """
        为数据集生成多个QA对（每个对话生成固定数量QA）。
        所有生成的QA都将直接管理在QACacheManager中。
        """
        if self.is_step:
            print("\n--- Step-by-step mode: QA Generation Preview. ---")
            print(f"\n您可以检查缓存文件 {self.cache_manager.current_cache_path} ，然后按回车键继续...")
            input("Press Enter to continue...")
            self.cache_manager.load_cache()
        # global_qa_idx should represent the count of currently 'exportable' (liked/generated) QAs
        # across the *entire* cache, for correct sequential indexing in the final output.
        global_qa_idx = len(self.cache_manager.get_exportable_qas())
        self.logger.info(f"Loaded {len(self.cache_manager.get_all_qas())} QAs from cache. Starting global QA index from {global_qa_idx}.")

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
                    qa_dict, _ = self._generate_single_qa(conversation, global_qa_idx)
                    if qa_dict:
                        generated_count_for_current_difficulty += 1
                        global_qa_idx += 1
                        self.logger.info(f"成功生成了第 {generated_count_for_current_difficulty}/{num_qa_for_difficulty} 个 '{self.difficulty}' 难度QA")
                    else:
                        self.logger.info(f"重新生成第{generated_count_for_current_difficulty}/{num_qa_for_difficulty} 个 '{self.difficulty}' 难度QA")
    
    def _generate_single_qa(self, conversation: Conversation, global_qa_idx: int) -> Tuple[Dict | None, List[Session]]:
        """
        生成单个QA对，并处理缓存逻辑和用户交互。
        返回生成的QA字典和所选会话列表，如果生成失败或用户拒绝则返回 (None, None)。
        """
        # 从conversation中随机选择会话
        session_count = random.randint(
            self.min_sessions,
            min(self.max_sessions, len(conversation.sessions))
        )
        selected_sessions = random.sample(conversation.sessions, session_count)
        selected_sessions.sort(key=lambda s: s.id)
        # Prepare context for LLM
        session_context = self._build_session_context(selected_sessions)
        
        # Get guidance QAs
        # positive examples (status="liked")
        preferred_qas = self.cache_manager.get_preferred_qas(self.difficulty)
        # negative examples (status="disliked")
        disliked_qas = self.cache_manager.get_disliked_qas(self.difficulty)

        additional_guidance = self._build_additional_guidance(
            preferred_qas=preferred_qas, 
            disliked_qas=disliked_qas
        )

        qa_response = self.generate_qa(session_context, additional_guidance)
        if not qa_response:
            self.logger.warning("LLM did not return a valid response.")
            return None, None
        
        qa_dict = self._parse_response(qa_response)
        if not qa_dict:
            self.logger.warning("Failed to parse LLM response into a QA dictionary.")
            return None, None
        
        # Add conversation-specific and global metadata
        qa_dict["conversation_id"] = conversation.id
        qa_dict["session_ids"] = [s.id for s in selected_sessions]
        qa_dict["difficulty"] = self.difficulty
        qa_dict["qa_id"] = self.cache_manager.generate_qa_id(qa_dict)
        qa_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        if self.is_step:
            print("\n--- Step-by-step mode: New QA Generated. ---")
            print(f"Question: {qa_dict.get('question_text')}")
            print(f"\n请检查这次生成的问题。输入 'y' 标记为【偏好问题】（保存进数据集）；输入 'n' 标记为【不喜欢】（保存进cache），并重新生成；输入 'r' 重新生成（不保存进cache）；按回车键标记为【已生成】（保存进数据集）。")
            char = input("输入 'y', 'n', 'r'或回车键继续...\n").strip().lower() # Read and standardize user input
            
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
            # Not in step-by-step mode, automatically add as 'generated'
            self.cache_manager.add_qa(qa_dict, status="generated")
            self.cache_manager.save_cache()
            self.logger.info(f"QA {qa_dict['qa_id']} added as 'generated'.")
            return qa_dict, selected_sessions

    def _build_session_context(self, sessions: List[Session]) -> str:
        """构建会话上下文表示（支持结构化数据）"""
        context = ""
        for session in sessions:
            context += f"### Session ID: {session.id}\n"
            
            if session.tables:
                self.logger.debug(f"会话 {session.id} 构建表格上下文")
                context += "Data Type: Structured Table\n"
                for idx, table in enumerate(session.tables):
                    context += f"Table {idx}:\n"
                    for row_idx, row in enumerate(table.rows):
                        identifier = row.get("股票简称") or row.get("股票代码") or f"Row {row_idx}"
                        context += f"  Entity: {identifier}\n"
                        for k, v in row.items():
                            context += f"    {k}: {v}\n"
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
        构建额外的指导信息，包含偏好问题、不偏好问题和所有已生成问题文本。
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
                guidance += f"  Good Example {idx + 1}:\n"
                guidance += f"    Question: {qa.get('question_text', 'N/A')}\n"
                guidance += f"    Answer: {qa.get('answer_text', 'N/A')}\n"
                guidance += f"    Evidence: {', '.join(qa.get('evidence', []))}\n"
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
                guidance += f"  Bad Example {idx + 1}:\n"
                guidance += f"    Question: {qa.get('question_text', 'N/A')}\n"
            guidance += "\n"
            
        return guidance

    def generate_qa(self, session_context: str, additional_guidance: str) -> str:
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

        if additional_guidance:
            prompt += additional_guidance
        self.logger.debug(f"Prompt for difficulty '{self.difficulty}': {prompt}")
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]
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

    def _parse_response(self, response: str) -> Dict:
        """解析LLM响应为结构化字典"""
        try:
            temp = json.loads(response.strip())
            if "question" in temp:
                temp["question_text"] = temp.pop("question")
            if "answer" in temp:
                temp["answer_text"] = temp.pop("answer")

            if isinstance(temp.get("answer_text"), str):
                match = re.search(r"^The answer is:\s*(-?[0-9]+(?:\.[0-9]+)?)", temp["answer_text"])
                if match:
                    temp["answer_text"] = float(match.group(1))
                else:
                    temp["answer_text"] = temp["answer_text"].replace("The answer is: ", "").strip()
            
            if isinstance(temp.get("evidence"), list):
                temp["evidence"] = [str(e).strip() for e in temp["evidence"]]
            elif isinstance(temp.get("evidence"), str):
                temp["evidence"] = [temp["evidence"].strip()]
            else:
                temp["evidence"] = ["Unknown evidence"]

            return temp
        except Exception as e:
            self.logger.error(f"解析响应时发生错误: {e}.")
            return None

def main():
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.environ['LOG_FILE'] = f"qa_gen_v1_{timestamp}.log"
        logger = setup_logging()
        
        base_parser = get_base_parser()
        parser = argparse.ArgumentParser(parents=[base_parser])
        parser = qa_generation_args(parser)
        args = parser.parse_args()
        
        logger.info(f"Starting QA generation V1 for: {args.input_data}")
        
        # 从命令行参数获取难度计数
        difficulty_counts: Dict[DifficultyLevel, int] = {
            "easy": args.easy,
            "medium": args.medium,
            "hard": args.hard
        }

        # 如果所有难度计数都为0
        if all(count == 0 for count in difficulty_counts.values()):
            logger.warning("No difficulty level specified. Please use --easy, --medium, or --hard to specify the number of questions.")
            exit(0)

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
            cache_dir=args.cache_dir
        )
        
        # 加载数据
        dataset = load_data(args.input_data)
        logger.info(f"数据集包含 {len(dataset.conversations)} 个对话")
        
        qa_generator.batch_generate(dataset, difficulty_counts) 
        
        # 从缓存中获取所有可导出的QA对
        results = qa_generator.cache_manager.get_exportable_qas()
        for idx, qa_item in enumerate(results):
            qa_item["qa_index"] = idx
        logger.info(f"成功从缓存中收集到 {len(results)} 个QA对准备导出")

    except Exception as e:
        logger.error(f"An error occurred during QA generation: {e}")
    finally:
        if 'results' not in locals():
            results = []
        os.makedirs(args.output_dir, exist_ok=True)
        temp = (os.path.splitext(os.path.basename(args.input_data))[0]).split("_")[0]
        filename = f"{temp}_qa_v1.json"
        output_path = os.path.join(args.output_dir, filename)
        save_results(results, output_path)
        
        logger.info(f"QA generation V1 complete. Output to: {output_path}")

if __name__ == "__main__":
    main()