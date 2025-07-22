from typing import Any, List, Dict, Tuple
import logging
import re
import math # Import math for isclose
from utils.data_struct import Evidence

logger = logging.getLogger(__name__)

class Validator:
    """答案与证据验证器"""
    def compare_answers(self, a1: Any, a2: Any) -> bool:
        try:
            num1 = float(a1)
            num2 = float(a2)
            return math.isclose(num1, num2, rel_tol=1e-9)
        except (ValueError, TypeError):
            return str(a1).strip() == str(a2).strip()

    def compare_evidence(
            self,
            llm_evidence: List[Evidence],
            sql_evidence: List[dict],
        ) -> bool:
        """
        直接比较两个证据列表是否完全一致。
        llm_evidence: [(code, sname, tdate, value, col), ...]
        sql_evidence: [{'code':..., 'sname':..., 'tdate':..., 'value':..., 'suffix':...}, ...]
        """
        logger.debug(f"LLM Evidence: {llm_evidence}")
        logger.debug(f"SQL Evidence: {sql_evidence}")
        if len(llm_evidence) != len(sql_evidence):
            return False

        def key(item: Evidence):
            return (item[0], item[1], item[2], item[4], item[3])

        for left, right in zip(
            sorted(llm_evidence, key=key), sorted(sql_evidence, key=key)
        ):
            if not (
                left[0] == right[0]
                and left[1] == right[1]
                and left[2] == right[2]
                and left[4] == right[4]
                and math.isclose(left[3], right[3], rel_tol=1e-9)
            ):
                return False
        return True