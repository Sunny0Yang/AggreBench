import sqlite3
from typing import List, Dict, Tuple, Any
import logging
import re
from .struct import Table

logger = logging.getLogger(__name__)

class SqlEngine:
    def __init__(self, db_name=":memory:"):
        """初始化SQL引擎，可连接到内存数据库或文件数据库"""
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.conn.row_factory = sqlite3.Row

    def _parse_and_standardize_value(self, value: Any) -> Any:
        """
        解析并标准化数值，处理货币和百分比单位。
        将所有货币转换为“元”，百分比转换为小数。
        """
        if not isinstance(value, str):
            return value
        original_value_str = value.strip()
        currency_match = re.match(r"^(-?\d+(\.\d+)?)\s*(元|万元|亿元)$", original_value_str)
        if currency_match:
            num_part = float(currency_match.group(1))
            unit = currency_match.group(3)
            if unit == "万元":
                return num_part * 10000.0
            elif unit == "亿元":
                return num_part * 100000000.0
            else:
                return num_part
        
        percentage_match = re.match(r"^(-?\d+(\.\d+)?)\s*%$", original_value_str)
        if percentage_match:
            num_part = float(percentage_match.group(1))
            return num_part / 100.0

        try:
            return int(original_value_str)
        except ValueError:
            try:
                return float(original_value_str)
            except ValueError:
                return original_value_str

    # Keep _infer_data_type as it's crucial for correct column types (INTEGER, REAL)
    def _infer_data_type(self, values: List[Any]) -> str:
        """
        根据列中的值推断最佳的SQL数据类型。
        优先级：INTEGER -> REAL -> TEXT
        """
        can_be_integer = True
        can_be_real = True

        for val in values:
            if val is None:
                continue
            
            standardized_val = self._parse_and_standardize_value(val)

            if can_be_integer:
                if not isinstance(standardized_val, int) or isinstance(standardized_val, bool):
                    can_be_integer = False
            
            if can_be_real:
                if not isinstance(standardized_val, (int, float)):
                    can_be_real = False
            
            if not can_be_integer and not can_be_real:
                return "TEXT"

        if can_be_integer:
            return "INTEGER"
        elif can_be_real:
            return "REAL"
        else:
            return "TEXT"

    def create_table_from_struct(self, tables: List[Table]):
        """根据Table结构列表在数据库中创建表并插入数据"""
        cursor = self.conn.cursor()
        for i, table in enumerate(tables):
            table_name = f"Table_{i}" 

            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Infer data types for each original column header
            column_defs: List[Tuple[str, str]] = [] # (original_header, inferred_type)
            for header in table.headers:
                column_values = [row.get(header) for row in table.rows]
                inferred_type = self._infer_data_type(column_values)
                column_defs.append((header, inferred_type)) 
            
            # Dynamic CREATE TABLE statement with inferred types, quoting original headers
            # Example: "股票代码" TEXT, "最新价" REAL, "资金流向[20231201]" REAL
            columns_sql_def = ", ".join(f'"{col_name}" {col_type}' for col_name, col_type in column_defs)
            create_table_sql = f"CREATE TABLE {table_name} ({columns_sql_def})"
            
            try:
                cursor.execute(create_table_sql)
                logger.info(f"成功创建表结构: {table_name} ({create_table_sql})")

                # Insert data
                if table.rows:
                    # Placeholders for values based on the number of columns
                    placeholders = ", ".join(["?"] * len(table.headers))
                    
                    # Columns list for INSERT statement, using original headers directly and quoting them
                    insert_columns = ", ".join(f'"{h}"' for h in table.headers)
                    insert_sql = f'INSERT INTO {table_name} ({insert_columns}) VALUES ({placeholders})'
                    
                    for row in table.rows:
                        values_for_insert = []
                        for h in table.headers:
                            raw_value = row.get(h)
                            processed_value = self._parse_and_standardize_value(raw_value)
                            values_for_insert.append(processed_value)

                        cursor.execute(insert_sql, values_for_insert)
                    
                    self.conn.commit()
                    logger.info(f"成功填充表: {table_name}，插入 {len(table.rows)} 条数据。")
                else:
                    logger.info(f"表 {table_name} 没有数据可插入。")
            
            except sqlite3.Error as e:
                logger.error(f"创建或填充表 {table_name} 失败: {e}")
                self.conn.rollback() 
            except Exception as e:
                logger.error(f"未知错误在创建或填充表 {table_name}: {e}")
                self.conn.rollback()

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """执行SQL查询并返回结果"""
        cursor = self.conn.cursor()
        try:
            cursor.execute(query)
            results = [dict(row) for row in cursor.fetchall()]
            logger.info(f"查询执行成功。结果数量: {len(results)}")
            return results
        except sqlite3.Error as e:
            logger.error(f"执行查询失败: '{query}' -> {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误在执行查询: '{query}' -> {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    example_table_data = {
        "headers": [
            "股票代码", "股票简称", "最新价", "最新涨跌幅",
            "资金流向[20231201]", "资金流向[20231204]", "资金流向[20231205]",
            "资金流向[20231212]", "资金流向[20231213]", "资金流向[20231214]"
        ],
        "rows": [
            {
                "股票代码": "300033.SZ",
                "股票简称": "同花顺",
                "最新价": "287.50元",
                "最新涨跌幅": "-6.69%",
                "资金流向[20231201]": "2.79亿元",
                "资金流向[20231204]": "5.70亿元",
                "资金流向[20231205]": "4814.18万元",
                "资金流向[20231212]": "-4611856.00元",
                "资金流向[20231213]": "-148584583.00元",
                "资金流向[20231214]": "-212768729.00元"
            },
            {
                "股票代码": "600519.SH",
                "股票简称": "贵州茅台",
                "最新价": "1700.00元",
                "最新涨跌幅": "0.50%",
                "资金流向[20231201]": "1.00亿元",
                "资金流向[20231204]": "0.80亿元",
                "资金流向[20231205]": "2000.00万元",
                "资金流向[20231212]": "1234567.00元",
                "资金流向[20231213]": "98765432.00元",
                "资金流向[20231214]": "100000000.00元"
            },
            {
                "股票代码": "000001.SZ",
                "股票简称": "平安银行",
                "最新价": "10.00元",
                "最新涨跌幅": "1.00%",
                "资金流向[20231201]": "0.5亿元",
                "资金流向[20231204]": "0.2亿元",
                "资金流向[20231205]": "500万元",
                "资金流向[20231212]": "5000000.00元",
                "资金流向[20231213]": "-1000000.00元",
                "资金流向[20231214]": "2000000.00元"
            }
        ]
    }

    table_instance = Table(**example_table_data)

    sql_engine = SqlEngine()

    print("\n--- Testing Table Creation and Data Insertion with Original Headers and Type Inference ---")
    try:
        sql_engine.create_table_from_struct([table_instance])
        
        print("\n--- Verifying Data and Aggregation ---")

        # Now, use the original header names directly in queries, but remember to quote them!
        stock_name_col = "股票简称"
        stock_price_col = "最新价"
        price_change_col = "最新涨跌幅"
        fund_flow_col_1 = "资金流向[20231201]"
        fund_flow_col_12 = "资金流向[20231212]"
        
        # Test SELECT query for individual values
        query_individual = f'SELECT "{stock_name_col}", "{stock_price_col}", "{price_change_col}" FROM Table_0 WHERE "{stock_name_col}" = "同花顺"'
        results_individual = sql_engine.execute_query(query_individual)
        print(f"\nQuery (individual): {query_individual}")
        print(f"Results: {results_individual}")
        assert results_individual[0][stock_price_col] == 287.50
        assert results_individual[0][price_change_col] == -0.0669

        # Test SUM on '最新价'
        query_sum_price = f'SELECT SUM("{stock_price_col}") AS total_price FROM Table_0'
        results_sum_price = sql_engine.execute_query(query_sum_price)
        print(f"\nQuery (SUM 最新价): {query_sum_price}")
        print(f"Results: {results_sum_price}")
        assert results_sum_price[0]["total_price"] == 1997.50

        # Test AVG on '最新涨跌幅'
        query_avg_change = f'SELECT AVG("{price_change_col}") AS avg_change FROM Table_0'
        results_avg_change = sql_engine.execute_query(query_avg_change)
        print(f"\nQuery (AVG 最新涨跌幅): {query_avg_change}")
        print(f"Results: {results_avg_change}")
        expected_avg_change = (-0.0669 + 0.0050 + 0.0100) / 3
        assert abs(results_avg_change[0]["avg_change"] - expected_avg_change) < 1e-9

        # Test SUM on '资金流向[20231201]'
        query_sum_fund_flow = f'SELECT SUM("{fund_flow_col_1}") AS total_fund_flow FROM Table_0'
        results_sum_fund_flow = sql_engine.execute_query(query_sum_fund_flow)
        print(f"\nQuery (SUM 资金流向[20231201]): {query_sum_fund_flow}")
        print(f"Results: {results_sum_fund_flow}")
        assert results_sum_fund_flow[0]["total_fund_flow"] == (2.79 * 1e8) + (1.00 * 1e8) + (0.5 * 1e8)

        # Test SUM on '资金流向[20231212]'
        query_sum_fund_flow_12 = f'SELECT SUM("{fund_flow_col_12}") AS total_fund_flow_12 FROM Table_0'
        results_sum_fund_flow_12 = sql_engine.execute_query(query_sum_fund_flow_12)
        print(f"\nQuery (SUM 资金流向[20231212]): {query_sum_fund_flow_12}")
        print(f"Results: {results_sum_fund_flow_12}")
        expected_sum_12 = -4611856.00 + 1234567.00 + 5000000.00
        assert results_sum_fund_flow_12[0]["total_fund_flow_12"] == expected_sum_12

    except Exception as e:
        logger.error(f"测试期间发生错误: {e}", exc_info=True)
    finally:
        sql_engine.close()
        print("\n所有测试完成，数据库连接已关闭。")