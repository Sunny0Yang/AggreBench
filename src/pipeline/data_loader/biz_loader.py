# src/pipeline/bizfin_loader.py

import json
import os
import re
import logging
from typing import List, Dict, Tuple, Any
from utils.data_struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset, Evidence
from utils.session_simulator import SessionSimulator
from utils.prompt_templates import PERSONA

logger = logging.getLogger(__name__)

class BizFinLoader:
    def __init__(self, model:str, max_turns:int, is_step:bool, cache_dir: str,
                 combine_size = 10, generate_pseudo_dialogue=True,
                 col_mapping: Dict[str, str] = None
                 ):
        self.logger = logger
        self.combine_size = combine_size
        self.session_simulator = SessionSimulator(model=model, max_turns=max_turns, is_step=is_step, cache_dir=cache_dir)
        self.generate_pseudo_dialogue = generate_pseudo_dialogue
        self.persona = PERSONA["financial"]
        
        self.error_set = []
        self.col_mapping = col_mapping or {
            "code": "股票代码",
            "sname": "股票简称",
            "tdate": "日期",
            "net_flow": "资金流向",
            "outflow": "资金流出",
            "change_percent": "涨跌幅",
            "low_af": "最低价_前复权",
            "volume": "成交量",
            "lowest": "最低价",
            "inflow": "资金流入",
            "dde_net_big_order": "dde大单净额",
            "amount": "成交额",
            "close_af": "收盘价_前复权",
            "net_big_order_buy": "主动买入大单净额",
            "open_af": "开盘价_前复权",
            "highest": "最高价",
            "turnover_rate": "换手率",
            "close": "收盘价",
            "amplitude": "振幅",
            "high_af": "最高价_前复权",
            "change_af": "涨跌_前复权",
            "net_inflow": "资金净流入额"
        }

    def _create_combined_conversation(self, samples: List[Dict], conversation_id: str) -> Conversation:
        """创建组合对话对象"""
        sessions = []
        session_counter = 1   # 会话计数器
        
        # 处理每个样本
        for sample_idx, sample in enumerate(samples):
            # 提取样本中的会话
            sample_session = self._extract_session(sample, conversation_id, session_counter)
            if sample_session:
                sessions.append(sample_session)
                session_counter += 1
        self.logger.debug(f"self.error_set: {set(self.error_set)}")
        return Conversation(
            conversation_id=conversation_id,
            speakers=["Assistant"],
            sessions=sessions,
        )

    def _extract_session(self, sample: Dict, conversation_id: str, start_index: int) -> Session:
        """从样本中提取会话并重新编号，进行表格规范化处理。"""
        
        messages = sample.get("messages", [])
        
        # 1. 提取所有原始宽格式表格数据
        all_raw_tables_data = []
        for msg in messages:
            if msg.get("role") == "user":
                for content in msg.get("content", []):
                    if content.get("type") == "text":
                        raw_tables_extracted = self._extract_tables(content["text"])
                        all_raw_tables_data.extend(raw_tables_extracted)
        
        if not all_raw_tables_data:
            self.logger.warning(f"No raw tables found for session in sample {sample.get('id', 'N/A')}. Skipping session.")
            return None

        # 2. 将原始 dict tables 转换为 Table 对象列表
        raw_table_objects = []
        for table_data in all_raw_tables_data:
            table = Table(
                headers=table_data["headers"],
                rows=table_data["rows"]
            )
            raw_table_objects.append(table)
        
        # 3. 规范化原始表格数据
        normalized_tables = self._normalize_tables(raw_table_objects)

        if not normalized_tables:
            self.logger.warning(f"No normalized tables generated from raw tables for sample {sample.get('id', 'N/A')}. Skipping session.")
            return None # Skip session if no normalized tables are created

        session_id = f"{conversation_id}_session_{start_index}"
        
        turns = []
        if self.generate_pseudo_dialogue:
            # 4. 从规范化后的表格生成证据
            evidences = self._table_to_evidences(normalized_tables)
            
            # 5. 生成对话回合
            dialog = self.session_simulator.generate_dialog(
                evidences=evidences,
                persona=self.persona
            )
            
            # 6. 转换为MultiModalTurn对象
            for turn_idx, turn in enumerate(dialog):
                turns.append(MultiModalTurn(
                    turn_id=f"{session_id}_turn_{turn_idx+1}",
                    speaker=turn["speaker"],
                    content=turn["content"],
                    evidence=turn.get("mentioned_evidence","")
                ))
        else:
            # 如果不需要伪对话，添加简单的表头信息（基于规范化后的表格）
            turns.append(MultiModalTurn(
                turn_id=f"{session_id}_title",
                speaker="Assistant",
                content=f"Session contains {len(normalized_tables)} normalized tables"
            ))
            
        session = Session(
            session_id=session_id,
            time="N/A",
            participants = ["User", "Assistant"] if self.generate_pseudo_dialogue else ["Assistant"],
            turns=turns,
            tables=normalized_tables # 存储规范化后的表格
        )
        
        return session

    def _extract_tables(self, text_content: str) -> List[Dict]:
        """从文本内容中提取表格数据"""
        tables = []
        
        # 表格模式：匹配以 | 开头的行
        table_pattern = r'\|(.+?)\|\n\|([\s\-:]+)\|(.+?)\n([\s\S]+?)(?=\n\n|\n###|$)'
        matches = re.findall(table_pattern, text_content, re.MULTILINE)
        
        for match in matches:
            # 提取表头
            headers = [h.strip() for h in match[0].split('|') if h.strip()]
            
            # 提取数据行
            rows = []
            for row_str in match[3].split('\n'):
                if not row_str.strip() or row_str.startswith('|---'):
                    continue
                cols = [c.strip() for c in row_str.split('|') if c.strip()]
                if len(cols) == len(headers):
                    rows.append(dict(zip(headers, cols)))
            
            if rows:
                tables.append({
                    "headers": headers,
                    "rows": rows
                })
        return tables

    def _normalize_tables(self, raw_table_objects: List[Table]) -> List[Table]:
        """
        将原始的宽格式表格转换为规范化的表格
        """
        from collections import defaultdict
        metric_groups = defaultdict(list)
        fund_flow_header_pattern = re.compile(r"(.*?)\[(\d{8})]")
        for raw_table in raw_table_objects:
            for row_dict in raw_table.rows:
                stock_code = row_dict.get(self.col_mapping.get("code"))
                stock_name = row_dict.get(self.col_mapping.get("sname"))
                
                if not stock_code or not stock_name:
                    self.logger.warning(f"跳过缺少 '股票代码' 或 '股票简称' 的行: {row_dict}")
                    continue

                for header, value_str in row_dict.items():
                    date_match = fund_flow_header_pattern.match(header)
                    if date_match:
                        chinese_metric = date_match.group(1)
                        date_str = date_match.group(2) 

                        metric_eng = self._reverse_map(chinese_metric)
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                        
                        standardized_value = self._parse_and_standardize_value(value_str)

                        if isinstance(standardized_value, (int, float)):
                            metric_groups[metric_eng].append({
                                "code": stock_code,
                                "sname": stock_name,
                                "tdate": formatted_date,
                                metric_eng: standardized_value
                            })
                        else:
                            self.logger.warning(f"无法标准化{metric_eng}值 '{value_str}' for {stock_name} on {formatted_date}. 原始值: {value_str}")
        normalized_tables = []
        for metric_eng, rows in metric_groups.items():
            normalized_tables.append(Table(
                headers=["code", "sname", "tdate", metric_eng],
                rows=rows
            ))
        return normalized_tables

    def _reverse_map(self, key: str) -> str:
        for k, v in self.col_mapping.items():
            if v == key:
                return k
        self.logger.warning(f"找不到映射关系: {key}")
        self.error_set.append(key)
        return key

    def _parse_and_standardize_value(self, value: Any) -> Any:
        """
        解析并标准化数值，处理货币和百分比单位。
        将所有货币转换为“million”，百分比转换为小数。
        """
        if not isinstance(value, str):
            return value

        original_value_str = value.strip()
        currency_pattern = re.compile(r'^(-?\d+(?:\.\d+)?)\s*(.*元)$')
        percentage_pattern = re.compile(r"^(-?\d+(\.\d+)?)%$")
        # Handle percentage
        percentage_match = percentage_pattern.match(original_value_str)
        if percentage_match:
            try:
                return float(percentage_match.group(1)) / 100.0
            except ValueError:
                self.logger.warning(f"无法解析百分比: {original_value_str}")
                return original_value_str

        # Handle currency
        currency_match = currency_pattern.match(original_value_str)
        if currency_match:
            try:
                num_part = float(currency_match.group(1))
                unit = currency_match.group(2)
                # 以“万元”为基准做转换
                unit_to_wan = {
                    "元": 1e-4,
                    "港元": 1e-4,
                    "美元": 8e-4,
                    "万元": 1.0,
                    "万港元": 1.0,
                    "万美元": 8.0,
                    "亿元": 1e4,
                    "亿港元": 1e4,
                    "亿美元": 8e4,
                }
                if unit in unit_to_wan:
                    return num_part * unit_to_wan[unit]
                else:
                    raise ValueError(f"Unsupported currency unit: {unit}")
            except ValueError as e:
                raise ValueError(f"Invalid currency value: {original_value_str}") from e

        try:
            return round(float(original_value_str),2)
        except ValueError:
            return original_value_str

    def _table_to_evidences(self, table_objects: List[Table]) -> List[Evidence]:
        """
        把任意规范化后的 Table → List[Evidence]
        Evidence = Tuple[code, sname, date, value, metric]
        """
        evidences: List[Evidence] = []

        for tbl in table_objects:
            required = {"code", "sname", "tdate"}
            headers_set = set(tbl.headers)
            # 过滤掉缺关键列的表
            if not required.issubset(headers_set):
                self.logger.warning(f"跳过缺少关键列的表: {tbl.headers}")
                continue

            # 剩下的列就是真正的指标列（可能不止一个）
            metric_cols = [h for h in tbl.headers if h not in required]

            for row in tbl.rows:
                code  = str(row.get("code",  "unknown"))
                sname = str(row.get("sname", "unknown"))
                date  = str(row.get("tdate", "None"))

                for metric in metric_cols:
                    try:
                        val = float(row.get(metric, 0.0))
                    except (TypeError, ValueError):
                        continue
                    evidences.append((code, sname, date, val, metric))

        return list(set(evidences))

    def load(self, file_path: str) -> ConversationDataset:
        """加载BizFinBench数据集文件并转换为ConversationDataset"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        logger.info(f"加载BizFinBench数据集: {file_path}")
        
        try:
            # 读取JSONL文件
            data_lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data_lines.append(json.loads(line))
            
            # 处理每个样本，将它们组合成对话
            conversations = []
            for group_idx in range(0, len(data_lines), self.combine_size):
                group_samples = data_lines[group_idx:group_idx+self.combine_size]
                if not group_samples:
                    continue
                    
                # 创建组合对话
                conversation_id = f"conv_{group_idx//self.combine_size+1}"
                conversation = self._create_combined_conversation(group_samples, conversation_id)
                conversations.append(conversation)
            
            # 创建数据集对象
            logger.info(f"成功加载 {len(conversations)} 个对话")
            return ConversationDataset(conversations=conversations)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e.msg} (位置: {e.pos})")
            raise
        except Exception as e:
            logger.error(f"加载数据集时出错: {str(e)}")
            raise

    @staticmethod
    def save(dataset: ConversationDataset, output_path: str):
        """将数据集保存到文件"""
        serialized = []
        for conversation in dataset.conversations:
            conv_data = {
                "conversation_id": conversation.id,
                "speakers": conversation.speakers,
                "sessions": []
            }
            for session in conversation.sessions:
                session_data = {
                    "session_id": session.id,
                    "time": session.time,
                    "participants": session.participants,
                    "turns": [{
                        "turn_id": turn.id,
                        "speaker": turn.speaker,
                        "content": turn.content,
                        "mentioned_evidence": turn.mentioned_evidence,
                    } for turn in session.turns],
                    "tables": [{
                        "headers": table.headers,
                        "rows": table.rows
                    } for table in session.tables],
                }
                conv_data["sessions"].append(session_data)
            serialized.append(conv_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        

def main():
    import argparse
    import time
    from utils.logger import setup_logging
    from utils.params import get_base_parser, data_loader_args
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ['LOG_FILE'] = f"biz_loader_{timestamp}.log"
    logger = setup_logging()
    
    # 参数解析
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser = data_loader_args(parser)
    args = parser.parse_args()

    loader = BizFinLoader(model=args.model, 
                          max_turns=args.max_turns, is_step= args.is_step,
                          cache_dir=args.cache_dir, combine_size=args.combine_size,
                          generate_pseudo_dialogue=args.generate_pseudo_dialogue)
    dataset = loader.load(args.input_data)
    
    # 保存处理后的数据集
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{os.path.splitext(os.path.basename(args.input_data))[0]}_processed.json"
    output_path = os.path.join(args.output_dir, filename)
    loader.save(dataset, output_path)
    logger.info(f"处理后的数据已保存至: {output_path}")

if __name__ == "__main__":
    main()