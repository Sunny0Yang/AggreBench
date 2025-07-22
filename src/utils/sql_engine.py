import sqlite3
from typing import List, Dict, Tuple, Any
import logging
import re
from .data_struct import Table, Evidence


class SqlEngine:
    def __init__(self, db_name: str = ":memory:"):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_table_from_struct(self, tables: List[Table]):
        cur = self.conn.cursor()
        # ---------- 1. 把 List[Table] 转成 List[Evidence] ----------
        evidence: List[Evidence] = []
        for tbl in tables:
            for row_dict in tbl.rows:
                code   = str(row_dict["code"])
                sname  = str(row_dict["sname"])
                tdate  = str(row_dict["tdate"])
                value  = float(row_dict["net_flow"] if "net_flow" in row_dict else row_dict["outflow"])
                suffix = "net_flow" if "net_flow" in row_dict else "outflow"
                evidence.append((code, sname, tdate, value, suffix))

        grouped: dict[str, List[Evidence]] = {}
        for row in evidence:
            grouped.setdefault(row[4], []).append(row)

        for tbl_suffix, rows in grouped.items():
            tbl_name = f"Table_{tbl_suffix}"
            cur.execute(f"DROP TABLE IF EXISTS {tbl_name}")
            cur.execute(
                f"""CREATE TABLE {tbl_name} (
                    code  TEXT,
                    sname TEXT,
                    tdate TEXT,
                    value REAL,
                    suffix TEXT
                )"""
            )
            self.logger.debug(f"Create table {tbl_name}")
            cur.executemany(
                f"INSERT INTO {tbl_name} (code, sname, tdate, value, suffix) VALUES (?,?,?,?,?)",
                [(r[0], r[1], r[2], r[3], r[4]) for r in rows],
            )
            self.logger.debug(f"Insert {len(rows)} rows into {tbl_name}")
        for tbl_suffix, rows in grouped.items():
            tbl_name = f"Table_{tbl_suffix}"
            cur.execute(f"SELECT * FROM {tbl_name}")
            rows_fetched = [dict(r) for r in cur.fetchall()]
            self.logger.debug(f"Table {tbl_name} content: {rows_fetched}")
        self.conn.commit()

    def execute_query(self, query: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(query)
        return [dict(r) for r in cur.fetchall()]

    def close(self):
        self.conn.close()