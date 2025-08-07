import sqlite3
from typing import List, Dict, Tuple, Any
import logging
from .data_struct import Table

class SqlEngine:
    def __init__(self, db_name: str = ":memory:"):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_table_from_struct(self, tables: List[Dict], domain: str = "financial"):
        """
        根据领域创建统一格式的表
        """
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS unified_data")
        
        # 创建统一表结构
        if domain == "financial":
            create_sql = """
            CREATE TABLE unified_data (
                code TEXT,
                sname TEXT,
                tdate TEXT,
                value REAL,
                metric TEXT
            )
            """
        elif domain == "medical":
            create_sql = """
            CREATE TABLE unified_data (
                patient_id TEXT,
                timestamp TEXT,
                variable_name TEXT,
                value REAL,
                table_type TEXT
            )
            """
        else:
            self.logger.error(f"Unsupported domain: {domain}")
            raise ValueError(f"Unsupported domain: {domain}")
        
        cur.execute(create_sql)
        self.logger.info(f"Created unified table for domain: {domain}")
        
        # 插入数据
        insert_data = []
        for table in tables:
            for row in table.rows:
                if domain == "financial":
                    # 金融数据格式: (code, sname, tdate, value, metric)
                    insert_data.append((
                        str(row.get("code", "")),
                        str(row.get("sname", "")),
                        str(row.get("tdate", "")),
                        float(row.get("value", 0.0)),
                        str(row.get("metric", ""))
                    ))
                elif domain == "medical":
                    # 医疗数据格式: (patient_id, timestamp, variable_name, value, table_type)
                    insert_data.append((
                        str(row.get("PatientID", "")),
                        str(row.get("time_event", "")),
                        str(row.get("variable_name", "")),
                        float(row.get("value", 0.0)),
                        str(table.table_type)
                    ))
        
        cur.executemany(
            "INSERT INTO unified_data VALUES (?, ?, ?, ?, ?)",
            insert_data
        )
        self.conn.commit()
        self.logger.info(f"Inserted {len(insert_data)} rows into unified_data")

    def execute_query(self, query: str) -> List[Dict]:
        """执行SQL查询并返回字典列表"""
        cur = self.conn.cursor()
        try:
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            self.logger.error(f"Error executing query: {query} - {str(e)}")
            return []
        finally:
            cur.close()

    def close(self):
        self.conn.close()