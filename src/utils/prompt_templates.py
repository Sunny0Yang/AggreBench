from typing import Dict

QA_GENERATION_PROMPTS: Dict[str, str] =({
    "structured_easy_template_en": """
### Task: Generate Simple Aggregative Query Question
Please generate a *simple, direct* aggregative query question based on the provided structured table data.
This question should require a single, straightforward aggregation (e.g., COUNT, SUM, AVG, MAX, MIN) focusing on one specific column.
### Available Information(values are in RMB million)
{session_context}

### Rules
1. Operation: exactly one aggregate function.  
2. Scope: single column + explicit date range (e.g. “Dec 1-7 2023”).  
3. Output JSON only:
{{
  "question": "...?",
  "answer": <float | int>,
  "evidence": [
    ["code", "sname", "YYYY-MM-DD", <value>, <metric>],
    ...
  ]
}}
4. Evidence must cover **all rows** used in the aggregation.  
5. `answer` is the raw numeric result.  
6. Ensure each evidence entry is a **5-tuple** with a **float** numeric value.

### Example
{{
  "question": "What was the total net capital flow for 同花顺 from Dec 1-7 2023?",
  "answer": -156442.27,
  "evidence": [
    ["300033.SZ", "同花顺", "2023-12-01", 279000000.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-04", 570000000.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-05", 48141800.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-06", 2435500.00, "net_flow"]
  ]
}}
""",
    
    "structured_medium_template_en": """
### Task: Generate Medium Difficulty Aggregative Query Question
Please generate an aggregative query question of *medium difficulty* based on the provided structured table data.
This question should involve either:
- A two-step aggregation (e.g., AVG requires SUM and COUNT implicitly).
- Aggregation on one column with simple filtering conditions.
- Simple comparison between two aggregated values from the same table.

### Available Information(values are in RMB million)
{session_context}

### Rules
1. Operation: MUST use a two-step aggregation (e.g., AVG implies SUM and COUNT) OR a single aggregation with simple filtering (e.g., COUNT WHERE X > Y)
2. Scope: Involve one column with 1-2 simple filtering conditions, or direct comparison of two aggregated results from the same table. MUST include explicit date range (e.g. “Dec 1-7 2023”) No complex joins across multiple implicit "tables".  
3. Output JSON only:
{{
  "question": "...?",
  "answer": <float | int>,
  "evidence": [
    ["code", "sname", "YYYY-MM-DD", <value>, <metric>],
    ...
  ]
}}
4. Evidence must cover **all rows** used in the aggregation.  
5. `answer` is the raw numeric result.  
6. Ensure each evidence entry is a **5-tuple** with a **float** numeric value..

### Example
{{
  "question": "What is the average daily net capital flow for 同花顺 from Dec 1-7 2023?",
  "answer": -22348.90,
  "evidence": [
    ["300033.SZ", "同花顺", "2023-12-01", -225000000.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-04", -110000000.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-05", -486858200.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-06", -378564500.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-07", -364000000.00, "net_flow"]
  ]
}}
""",
    
    "structured_hard_template_en": """
### Task: Generate Complex Aggregative Query Question  
Please generate a *complex, multi-step* aggregative query question based on the provided structured table data.
This question should require:
- Multiple nested aggregations or complex calculations across different columns.
- Sophisticated filtering, potentially involving multiple conditions or implied relationships.
- Time-based analysis or comparisons (e.g., year-over-year growth, trends over periods).
- Potentially combine information from multiple conceptual "tables" if the context implies them (e.g., different sections of the data representing different entities or timeframes that need to be linked).
---

### Available Information(values are in RMB million)
{session_context}

### Rules
1. Operation: exactly one aggregate function.  
2. Scope: single column + explicit date range (e.g. “Dec 1-7 2023”).  
3. Output JSON only:
{{
  "question": "...?",
  "answer": <float | int>,
  "evidence": [
    ["code", "sname", "YYYY-MM-DD", <value>, <metric>],
    ...
  ]
}}
4. Evidence must cover **all rows** used in the aggregation.  
5. `answer` is the raw numeric result.  
6. Ensure each evidence entry is a **5-tuple** with a **float** numeric value.

### Example
{{
  "question": "Calculate the percentage change in total net capital flow for 同花顺 from Dec 1-7 to Dec 22-28 2023 (rounded to two decimals).",
  "answer": 118.68,
  "evidence": [
    ["300033.SZ", "同花顺", "2023-12-01", -225000000.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-04", -110000000.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-05", -486858200.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-06", -378564500.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-07", -364000000.00, "net_flow"],
    ["300033.SZ", "同花顺", "2023-12-22", -342013070.00, "net_flow"]
  ]
}}
""",

"sql_prompt_template": """
You are an advanced AI assistant specializing in generating precise SQL queries based on natural language questions and provided table schemas and data. Your task is to generate two distinct SQL queries: one to retrieve the **answer** to the given question and another to identify the **evidence** supporting that answer, both from the provided tables.

---

### **Instructions:**

1.  **Analyze the Question:** Carefully understand the user's question to identify the specific information being requested.
2.  **Examine the Tables:** Review the provided table schemas and their sample data to determine which tables and columns are relevant to answer the question and find supporting evidence.
3.  **Generate SQL Queries (`SQL_ANSWER` and `SQL_EVIDENCE`):**
    * Construct syntactically correct SQL queries that accurately reference the provided tables and columns.
    * **Crucial for Column Names:** If a column name contains **special characters** (e.g., `[`, `]`, ` ` (space), `-`), or if it's a **reserved keyword** in SQL, you **MUST** enclose it in **double quotes** (e.g., `"Column Name With Spaces"`, `"资金流向[20231201]"`). This applies to all clauses (SELECT, FROM, WHERE, etc.).
    * Prioritize queries that return the most concise and direct answer.
    * Use appropriate SQL clauses (e.g., `SELECT`, `FROM`, `WHERE`, `JOIN`, `GROUP BY`, `ORDER BY`, `WITH`, `HAVING`, aggregate functions) as needed.
    * **SQLite Compatibility Note (CRITICAL):** You are generating SQL for **SQLite3**, and ensure all mathematical and logical expressions are correctly parenthesized to avoid syntax errors.
4.  **Specifics for Answer SQL Query (`SQL_ANSWER`):**
    * This query should directly yield the answer to the user's question.

5.  **Specifics for Evidence SQL Query (`SQL_EVIDENCE`):**
    * This query should retrieve the data points from the tables that serve as **direct evidence** for the answer.
    * For each row selected as evidence, ensure you `SELECT` an entire row.
    * This query should ideally return the foundational facts or figures that lead to the answer.
    * Consider returning relevant columns from the rows that contain the key information.

---
### **Table Information:**

{tables}

---

### **Question:**

{question}

---

### **Output Format:**

Provide your response in the following format. Ensure there are no additional explanations or text, only the two SQL queries.

```sql
SQL_ANSWER:
{{sql_text}};

SQL_EVIDENCE:
{{sql_text}};
"""
})

PERSONA: Dict[str, str] = ({
    "financial":"""
This persona represents a **prudent individual investor** with a foundational understanding of financial markets and personal finance principles. A primary focus involves monitoring **macroeconomic trends, corporate financial reports, and the inherent risks of investment products**. Investment decisions are typically preceded by **thorough research and the seeking of professional counsel**.

Areas of interest for this persona include a range of financial instruments such as **stocks, mutual funds, bonds, real estate, and retirement planning vehicles**. The overarching objective is to achieve **stable asset growth and effective risk management**, rather than pursuing short-term speculative gains.

The communication style tends to be **pragmatic and inquisitive**, frequently involving the posing of specific financial questions or requests for investment guidance.
"""
})

SESSION_SIMULATOR_PROMPT: Dict[str, str] = ({
    "user": """
You are acting as a normal user chatting with a professional AI assistant about financial data.

Your persona is:
{persona}

Your overarching goal is to ensure a **complete and thorough discussion** of **ALL** the financial data that still needs to be covered.

Remaining **Un-discussed Financial Data** for this session (values are in RMB million):
{evidences}

---

Decision-making process
1. Examine the `Remaining Un-discussed Financial Data` list provided.
2. **Choose a Strategy:**
  * **Option 1 (Present Data + Ask for Analysis): From the list, choose around 8 semantically related data points, present them clearly and naturally, and then pose a real-world meaningful question for analysis.
  * **Option 2 (Query for Specific Time Period):** Formulate a specific question asking for data within a defined date range (e.g., "December 1st to December 10th, 2023"). 
3. Tone: neutral, casual, not overly polite.
4. **After your message**, list every **exact original tuple evidence** you just explicitly implicated. Each evidence MUST be on a new line, starting with '- ', and contain ONLY ONE complete tuple. This `EVIDENCES_USED_IN_THIS_TURN:` block is NOT part of chat history.

Last Assistant Response:
{last_turn_content}

### Example 
Example for Option 1 (present data + ask for analysis):
"I've been looking at 同花顺 (300033.SZ)'s capital flow data for December 2023. For example, on Dec 1st, there was an inflow of RMB 279 million, followed by RMB 570 million on Dec 4th, and RMB 456 million on Dec 8th. However, the trend shifted, with an outflow of RMB 148.58 million on Dec 13th, and RMB 212.77 million on Dec 14th. What's your analysis of these fluctuations and their potential impact?"

EVIDENCES_USED_IN_THIS_TURN:
- ('300033.SZ', '同花顺', '2023-12-01', 279.0, 'net_flow')
- ('300033.SZ', '同花顺', '2023-12-04', 570.0, 'net_flow')
- ('300033.SZ', '同花顺', '2023-12-08', 456.0, 'net_flow')
- ('300033.SZ', '同花顺', '2023-12-13', -148.58, 'net_flow')
- ('300033.SZ', '同花顺', '2023-12-14', -212.77, 'net_flow')

Example for Option 2 (query for specific time period):
"Can you give me the daily net inflow for 同花顺 (300033.SZ) during the first ten days of December 2023?"

EVIDENCES_USED_IN_THIS_TURN:
none
for Option 2 (This section would be empty as the user is *requesting* new data, not referencing existing data from the list):
""",

    "assistant": """
You are a professional AI assistant specialized in finance. 
Your goal is to answer concisely and **ensure every remaining data point is eventually discussed**.

Remaining **Un-discussed Financial Data**: 
{evidences}

---

Decision-making process 
1. **If the user supplied data** → analyse directly. 
2. **If the user requested data** → retrieve and present the relevant **tuple evidences** from the list. 
3. **After answering**, proactively surface any **still-un-discussed tuple evidences** when natural. 
4. Keep tone professional and succinct.

Last Turn: 
{last_turn_content}

Example:
When user ask: "Can you give me the daily net inflow for 同花顺 (300033.SZ) during the first ten days of December 2023?"

You should response(remember that all the data are from the list "Remaining **Un-discussed Financial Data**", if there is no data in the list you needed, you should response "Sorry, I don't have the data for that time period."):
Sure, here are the daily net inflow for 同花顺 (300033.SZ) during the first three days of December 2023:
- Dec 1: RMB 279 million
- Dec 2: RMB 570 million
- Dec 3: RMB 456 million

EVIDENCES_USED_IN_THIS_TURN:
- ('300033.SZ', '同花顺', '2023-12-01', 279.0, 'net_flow')
- ('300033.SZ', '同花顺', '2023-12-02', 570.0, 'net_flow')
- ('300033.SZ', '同花顺', '2023-12-03', 456.0, 'net_flow')
"""
})