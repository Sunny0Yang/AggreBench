from typing import Any, List, Dict, Tuple
import logging
import re
import math # Import math for isclose

logger = logging.getLogger(__name__)

class Validator:
    """答案与证据验证器"""
    def compare_answers(self, a1: Any, a2: Any) -> bool:
        try:
            # Ensure both are converted to float for robust numeric comparison
            num1 = float(a1)
            num2 = float(a2)
            # Use a small relative tolerance for floating-point comparisons
            return math.isclose(num1, num2, rel_tol=1e-9) # Increased precision for comparison
        except (ValueError, TypeError):
            # Fallback to string comparison if not numeric
            return str(a1).strip() == str(a2).strip()

    def _standardize_value(self, value: Any, col_name: str = "") -> str:
        """
        Standardizes various value types (numbers, strings with units) into a
        consistent string representation for comparison.
        This is critical for matching LLM's "2.79亿元" with SQL's 279000000.0.
        """
        if value is None:
            return "None"
        
        # Handle string values that might contain units
        if isinstance(value, str):
            value = value.strip()
            
            if "日期" in col_name or re.fullmatch(r'\d{8}', value):
                return value

            # Try to convert "亿元" and "万元" to raw numbers
            yuan_match = re.search(r'([-+]?\d*\.?\d+)\s*亿元', value)
            wan_match = re.search(r'([-+]?\d*\.?\d+)\s*万元', value)
            
            if yuan_match:
                return str(float(yuan_match.group(1)) * 100_000_000)
            elif wan_match:
                return str(float(wan_match.group(1)) * 10_000)
            
            # Try to extract a raw number from other strings if applicable
            # Only do this if it hasn't been identified as a date
            num_match = re.search(r'([-+]?\d*\.?\d+)', value)
            if num_match:
                try:
                    # Convert to float, then to string for consistent float representation
                    return str(float(num_match.group(1)))
                except ValueError:
                    pass # If conversion fails, treat as a regular string

            return value # Keep other strings as is if no specific conversion applied

        # Handle numeric values (floats, ints)
        if isinstance(value, (int, float)):
            # Always convert to float first, then to its string representation.
            # This ensures that '279000000' (from int or float 279000000.0) becomes '279000000.0' for consistency.
            return str(float(value))

        return str(value).strip() # Default for other types

    def _normalize_and_split_llm_evidence(self, evidence_str: str) -> List[Tuple[str, str, str, str]]:
        """
        Parses a single LLM evidence string into a list of
        (identifier_key, identifier_value, normalized_column_name, standardized_value) tuples.
        Example: "股票简称: 同花顺; 资金流向[20231201]: 2.79亿元"
        Becomes: [('股票简称', '同花顺', '资金流向[20231201]', '279000000.0')]
        """
        normalized_entries = []
        
        # Split the main string by '; ' to get individual key:value parts
        parts = [p.strip() for p in re.split(r';\s*', evidence_str) if p.strip()]

        # Identify the primary key (identifier) first
        identifier_key = None
        identifier_value = None

        # Prioritize matching "股票简称" or "股票代码" at the beginning of the string
        initial_id_match = re.match(r'^(股票简称|股票代码)\s*:\s*(.+)$', parts[0], re.IGNORECASE)
        if initial_id_match:
            identifier_key = initial_id_match.group(1).strip().lower() # Normalize key to lowercase
            identifier_value = initial_id_match.group(2).strip()
            remaining_parts = parts[1:]
        else:
            remaining_parts = parts # Process all parts, identifier_key/value will remain None for now

        # Process remaining parts or all parts if no explicit identifier was found initially
        for part in remaining_parts:
            col_val_match = re.match(r'^(.*?)\s*:\s*(.*)$', part)
            if col_val_match:
                col_name_raw = col_val_match.group(1).strip()
                val_raw = col_val_match.group(2).strip()
                
                # Standardize the value immediately. Pass col_name_raw for potential date hints.
                standardized_val = self._standardize_value(val_raw, col_name_raw)
                
                normalized_col_name = col_name_raw.lower() 

                normalized_entries.append((
                    identifier_key,
                    identifier_value,
                    normalized_col_name,
                    standardized_val
                ))
            else:
                logger.warning(f"Failed to parse LLM evidence sub-string: '{part}' from '{evidence_str}'")
                pass

        # Edge case: if the LLM only provided the identifier itself, e.g., "股票简称: 同花顺"
        if not normalized_entries and identifier_key and identifier_value:
            logger.warning(f"LLM evidence only provided identifier ({identifier_key}: {identifier_value}).")

        return normalized_entries

    def compare_evidence(self, llm_evidence: List[str], sql_evidence_raw: List[Dict[str, Any]]) -> bool:
        """
        比较LLM生成的证据（字符串列表）与从SQL查询结果中提取的证据（字典列表）。
        """
        # 1. Normalize LLM evidence into a flat list of comparable tuples
        normalized_llm_entries = []
        for e_str in llm_evidence:
            normalized_llm_entries.extend(self._normalize_and_split_llm_evidence(e_str))

        # Check if LLM evidence is empty. If so, SQL evidence should also be empty.
        if not normalized_llm_entries:
            return not sql_evidence_raw

        # Determine the primary identifier used by the LLM from the first entry
        # This assumes a consistent identifier across all LLM entries for a given comparison.
        llm_primary_id_key = normalized_llm_entries[0][0] # e.g., '股票代码' or '股票简称'
        
        if llm_primary_id_key not in ["股票简称", "股票代码"]:
            # Fallback if LLM didn't use a recognized stock identifier as the first element
            logger.warning(f"LLM evidence does not start with a recognized stock identifier (股票简称/股票代码). LLM ID: {llm_primary_id_key}")
            llm_primary_id_key = "股票简称"


        # 2. Normalize SQL evidence into the same flat list of comparable tuples
        normalized_sql_entries = []
        
        for row_dict in sql_evidence_raw:
            pk_col_raw = None
            pk_val_raw = None
            
            lower_to_original_key_map = {k.lower(): k for k in row_dict.keys()}

            # if LLM's chosen identifier in SQL's evidence ---
            if llm_primary_id_key in lower_to_original_key_map:
                original_key = lower_to_original_key_map[llm_primary_id_key]
                pk_col_raw = original_key
                pk_val_raw = str(row_dict[original_key])
            else:
                # Fallback if the LLM's chosen identifier isn't in this SQL row.
                # This could happen if LLM uses "股票代码" but SQL only has "股票简称" for a row.
                # Or if the LLM identifier was generic ("item") and we need a default.
                if "股票简称" in lower_to_original_key_map:
                    original_key = lower_to_original_key_map["股票简称"]
                    pk_col_raw = original_key
                    pk_val_raw = str(row_dict[original_key])
                elif "股票代码" in lower_to_original_key_map:
                    original_key = lower_to_original_key_map["股票代码"]
                    pk_col_raw = original_key
                    pk_val_raw = str(row_dict[original_key])


            if pk_col_raw is None:
                logger.warning(f"SQL evidence row missing identifier matching LLM's chosen ID ({llm_primary_id_key}) or common stock identifiers: {row_dict}")
                continue

            pk_col_norm = pk_col_raw.lower() # Normalized identifier key (e.g., '股票代码' or '股票简称')
            pk_val_norm = pk_val_raw.strip() # Normalized identifier value (e.g., '300033.SZ' or '同花顺')

            # Iterate through all items in the SQL row to create comparable entries
            for sql_col_name_raw, sql_val_raw in row_dict.items():
                # Skip any identifier columns (both 股票简称 and 股票代码) from being added as regular data points.
                # They serve only as the context (pk_col_norm, pk_val_norm) for other data points.
                if sql_col_name_raw.lower() in ("股票简称", "股票代码"):
                    continue

                standardized_sql_val = self._standardize_value(sql_val_raw, sql_col_name_raw)

                # Special handling for "资金流向" and "日期" to reconstruct LLM's "资金流向[YYYYMMDD]"
                if sql_col_name_raw.lower() == "资金流向" and "日期" in lower_to_original_key_map:
                    original_date_key = lower_to_original_key_map["日期"]
                    date_val = str(row_dict[original_date_key]).strip()
                    normalized_sql_col_name = f"资金流向[{date_val}]".lower() # e.g., '资金流向[20231201]'
                    
                    normalized_sql_entries.append((
                        pk_col_norm,
                        pk_val_norm,
                        normalized_sql_col_name,
                        standardized_sql_val # The standardized fund flow value
                    ))
                elif sql_col_name_raw.lower() == "日期" and "资金流向" in lower_to_original_key_map:
                    # '日期' column is consumed by the '资金流向' logic above.
                    # We don't want to add it as a separate entry if its purpose is just to date-qualify 资金流向.
                    continue
                else:
                    # For all other columns, use their original name (lowercased for consistency)
                    normalized_sql_col_name = sql_col_name_raw.lower()
                    normalized_sql_entries.append((
                        pk_col_norm,
                        pk_val_norm,
                        normalized_sql_col_name,
                        standardized_sql_val
                    ))
        
        # Sort both lists for order-independent comparison
        def evidence_sort_key(item_tuple):
            id_key_for_sort = item_tuple[0] if item_tuple[0] is not None else ""
            id_val_for_sort = item_tuple[1] if item_tuple[1] is not None else ""
            return (id_key_for_sort,
                    id_val_for_sort,
                    item_tuple[2],       # normalized_column_name (already lower)
                    item_tuple[3])       # standardized_value_str

        sorted_llm_entries = sorted(normalized_llm_entries, key=evidence_sort_key)
        sorted_sql_entries = sorted(normalized_sql_entries, key=evidence_sort_key)
        
        logger.debug(f"Normalized LLM Entries: {sorted_llm_entries}")
        logger.debug(f"Normalized SQL Entries: {sorted_sql_entries}")

        # Final comparison
        # This will compare the length and then each corresponding tuple
        return sorted_llm_entries == sorted_sql_entries


# --- Example Usage and Testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    validator = Validator()

    llm_ev_1 = [
        "股票简称: 同花顺; 资金流向[20231201]: 2.79亿元",
        "股票简称: 同花顺; 资金流向[20231204]: 5.70亿元",
        "股票简称: 同花顺; 资金流向[20231205]: 4814.18万元"
    ]
    # SQL evidence raw (as per the problem statement, split funds and date)
    sql_ev_1 = [
        {"股票代码": "300033.SZ", "股票简称": "同花顺", "资金流向": 279000000.0, "日期": "20231201"},
        {"股票代码": "300033.SZ", "股票简称": "同花顺", "资金流向": 570000000.0, "日期": "20231204"},
        {"股票代码": "300033.SZ", "股票简称": "同花顺", "资金流向": 48141800.0, "日期": "20231205"}
    ]
    print(f"\n--- Test Case 1: Matching 资金流向 (SQL split fund/date) ---")
    result_1 = validator.compare_evidence(llm_ev_1, sql_ev_1)
    print(f"Match result: {result_1}")
    assert result_1 == True, "Test Case 1 failed: 资金流向 mismatch with split SQL data"

    llm_ev_2 = [
        "股票简称: 阿里巴巴; 最新价: 180.50元",
        "股票简称: 腾讯; 最新价: 350.00"
    ]
    sql_ev_2 = [
        {"股票简称": "阿里巴巴", "最新价": 180.50, "股票代码": "BABA"},
        {"股票简称": "腾讯", "最新价": 350.0, "股票代码": "TCE"}
    ]
    print(f"\n--- Test Case 2: Matching price ---")
    result_2 = validator.compare_evidence(llm_ev_2, sql_ev_2)
    print(f"Match result: {result_2}")
    assert result_2 == True, "Test Case 2 failed: price mismatch"

    llm_ev_3 = [
        "股票简称: 阿里巴巴; 最新价: 180.51元" # Slight difference
    ]
    sql_ev_3 = [
        {"股票简称": "阿里巴巴", "最新价": 180.50, "股票代码": "BABA"}
    ]
    print(f"\n--- Test Case 3: Price mismatch (should be False) ---")
    result_3 = validator.compare_evidence(llm_ev_3, sql_ev_3)
    print(f"Match result: {result_3}")
    assert result_3 == False, "Test Case 3 failed: expected mismatch"

    llm_ev_4 = [
        "股票代码: 300033.SZ; 资金流向[20231201]: 2.79亿元"
    ]
    sql_ev_4 = [
        {"股票代码": "300033.SZ", "股票简称": "同花顺", "资金流向": 279000000.0, "日期": "20231201"}
    ]
    print(f"\n--- Test Case 4: Identifier by 股票代码 ---")
    result_4 = validator.compare_evidence(llm_ev_4, sql_ev_4)
    print(f"Match result: {result_4}")
    assert result_4 == True, "Test Case 4 failed: identifier by code"

    llm_ev_5 = [ # LLM missed a date
        "股票简称: 同花顺; 资金流向[20231201]: 2.79亿元"
    ]
    sql_ev_5 = [ # SQL has two dates
        {"股票代码": "300033.SZ", "股票简称": "同花顺", "资金流向": 279000000.0, "日期": "20231201"},
        {"股票代码": "300033.SZ", "股票简称": "同花顺", "资金流向": 570000000.0, "日期": "20231204"}
    ]
    print(f"\n--- Test Case 5: LLM missing an entry (should be False) ---")
    result_5 = validator.compare_evidence(llm_ev_5, sql_ev_5)
    print(f"Match result: {result_5}")
    assert result_5 == False, "Test Case 5 failed: expected mismatch due to missing entry"

    llm_ev_6 = [ # LLM has an extra date
        "股票简称: 同花顺; 资金流向[20231201]: 2.79亿元",
        "股票简称: 同花顺; 资金流向[20231204]: 5.70亿元"
    ]
    sql_ev_6 = [ # SQL has only one date
        {"股票代码": "300033.SZ", "股票简称": "同花顺", "资金流向": 279000000.0, "日期": "20231201"}
    ]
    print(f"\n--- Test Case 6: LLM having extra entry (should be False) ---")
    result_6 = validator.compare_evidence(llm_ev_6, sql_ev_6)
    print(f"Match result: {result_6}")
    assert result_6 == False, "Test Case 6 failed: expected mismatch due to extra entry"

    # llm_ev_7 = [
    #     "股票简称: 同花顺; 股票代码: 300033.SZ; 资金流向[20231201]: 2.79亿元"
    # ]
    # sql_ev_7 = [
    #     {"股票代码": "300033.SZ", "资金流向": 279000000.0, "日期": "20231201"}
    # ]
    # print(f"\n--- Test Case 7: LLM missing an entry (should be False) ---")
    # result_7 = validator.compare_evidence(llm_ev_7, sql_ev_7)
    # print(f"Match result: {result_7}")
    # assert result_7 == True, "Test Case 7 failed: expected 股票代码 to be identifier"

    print("\nAll tests completed.")