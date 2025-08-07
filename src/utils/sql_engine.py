import sqlite3
from typing import List, Dict, Tuple, Any
import logging
import re
from .data_struct import Table


class SqlEngine:
    def __init__(self, db_name: str = ":memory:"):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_table_from_struct(self, tables: List[Table]):
        cur = self.conn.cursor()
        # 1) 构建 Evidence (code, sname, tdate, value, metric)
        evidence: List[Tuple] = []
        reserved = {"code", "sname", "tdate"}
        for tbl in tables:
            metric_cols = [h for h in tbl.headers if h not in reserved]
            for row in tbl.rows:
                code  = str(row.get("code", ""))
                sname = str(row.get("sname", ""))
                tdate = str(row.get("tdate", ""))
                for metric in metric_cols:
                    try:
                        val = float(row[metric])
                    except (TypeError, ValueError, KeyError):
                        continue
                    evidence.append((code, sname, tdate, val, metric))
        # 2) 按 metric 建表 & 插入        
        grouped: dict[str, List[Tuple]] = {}
        for row in evidence:
            grouped.setdefault(row[4], []).append(row)

        for metric, rows in grouped.items():
            tbl_name = f"Table_{metric}"
            cur.execute(f"DROP TABLE IF EXISTS {tbl_name}")
            self.logger.debug(f"Dropped table {tbl_name}")
            cur.execute(
                f"""CREATE TABLE {tbl_name} (
                    code  TEXT,
                    sname TEXT,
                    tdate TEXT,
                    value REAL
                )"""
            )
            self.logger.debug(f"Created table {tbl_name}")
            cur.executemany(
                f"INSERT INTO {tbl_name} (code, sname, tdate, value) VALUES (?,?,?,?)",
                [(r[0], r[1], r[2], r[3]) for r in rows],
            )
            self.logger.debug(f"Inserted {len(rows)} rows into {tbl_name}")
        self.conn.commit()

    def execute_query(self, query: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(query)
        return [dict(r) for r in cur.fetchall()]

    def close(self):
        self.conn.close()