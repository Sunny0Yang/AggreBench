from typing import Any, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class Validator:
    """答案与证据验证器"""
    
    def compare_answers(self, a1: Any, a2: Any) -> bool:
        try:
            num1 = float(a1)
            num2 = float(a2)
            return abs(num1 - num2) < 1e-5
        except (ValueError, TypeError):
            return str(a1).strip() == str(a2).strip()

    def compare_evidence(
            self,
            llm_evidence: List[Dict],
            sql_evidence: List[Dict],
            domain: str = "financial"
        ) -> bool:
        """
        比较两个证据列表是否一致（忽略顺序）
        """
        if len(llm_evidence) != len(sql_evidence):
            return False
        
        # 转换证据为可比较的格式
        llm_set = set()
        sql_set = set()
        
        if domain == "financial":
            for item in llm_evidence:
                llm_set.add((
                    item.get("code", ""),
                    item.get("sname", ""),
                    item.get("tdate", ""),
                    round(float(item.get("value", 0)), 5),
                    item.get("metric", "")
                ))
            for item in sql_evidence:
                sql_set.add((
                    item.get("code", ""),
                    item.get("sname", ""),
                    item.get("tdate", ""),
                    round(float(item.get("value", 0)), 5),
                    item.get("metric", "")
                ))
        elif domain == "medical":
            for item in llm_evidence:
                llm_set.add((
                    item.get("patient_id", ""),
                    item.get("timestamp", ""),
                    item.get("variable_name", ""),
                    round(float(item.get("value", 0)), 5),
                    item.get("table_type", "")
                ))
            for item in sql_evidence:
                sql_set.add((
                    item.get("patient_id", ""),
                    item.get("timestamp", ""),
                    item.get("variable_name", ""),
                    round(float(item.get("value", 0)), 5),
                    item.get("table_type", "")
                ))
        else:
            for item in llm_evidence:
                llm_set.add((
                    item.get("entity_id", ""),
                    item.get("timestamp", ""),
                    item.get("variable_name", ""),
                    round(float(item.get("value", 0)), 5),
                    item.get("table_type", "")
                ))
            for item in sql_evidence:
                sql_set.add((
                    item.get("entity_id", ""),
                    item.get("timestamp", ""),
                    item.get("variable_name", ""),
                    round(float(item.get("value", 0)), 5),
                    item.get("table_type", "")
                ))
        
        return llm_set == sql_set