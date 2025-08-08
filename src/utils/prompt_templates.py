from typing import Dict
# 在utils/prompt_templates.py中
SYSTEM_PROMPTS = {
    "financial": "You are a financial data analyst specializing in generating aggregation queries on structured financial tables. Your questions should focus on financial metrics like capital flows, stock performance, and market trends.",

    "medical": "You are a medical data analyst with clinical expertise. Generate scientifically meaningful questions based on structured medical data. Frame questions in the context of patient care and clinical decision-making.",
}
QA_GENERATION_PROMPTS = ({
    # 金融领域 - 简单难度（单次聚合）
    "financial_structured_easy_template_en": """
### Task: Generate Financial Query with SQL
Generate a question that requires a single aggregation (SUM, AVG, COUNT, MIN, MAX) 
over a small set of values from the structured financial data and corresponding SQL queries.

### Available Information (values in RMB million)
{session_context}

### Rules
1. **Question Generation Rules**:
   - Question must involve exactly one aggregation function
   - Scope limited to a single entity (stock) and short time range (1-3 days)
   - No complex filtering conditions
   - Question MUST clearly specify the stock name and code

2. **SQL Generation Rules**:
   - Use EXACT variable names from the session context
   - Ensure SQLite compatibility(sqlite3)
   - SQL_ANSWER query must directly yield the answer to the question
   - SQL_EVIDENCE query must retrieve all rows used to yield the answer
   - Both queries must be syntactically correct and executable

3. **Data-Specific Rules**:
   - `code`: Stock code (e.g., "300033.SZ")
   - `sname`: Stock name (e.g., "Tonghuashun")
   - `tdate`: Transaction date (format: "YYYY-MM-DD")
   - `value`: Numeric value (in RMB million)
   - `metric`: Financial metric (e.g., "net_flow", "volume")

4. Output JSON only:
{{
  "question": "...?",
  "answer": <float | int>,
  "evidence": [
    ["code", "sname", "YYYY-MM-DD", <value>, <metric>],
    ...
  ],
  "sql_answer_query": "SELECT ...",
  "sql_evidence_query": "SELECT ..."
}}

### Example
{{
  "question": "What was the average daily net capital flow for Tonghuashun (300033.SZ) from December 1-3, 2023?",
  "answer": 185.0,
  "evidence": [
    ["300033.SZ", "Tonghuashun", "2023-12-01", 279.0, "net_flow"],
    ["300033.SZ", "Tonghuashun", "2023-12-02", 150.0, "net_flow"],
    ["300033.SZ", "Tonghuashun", "2023-12-03", 126.0, "net_flow"]
  ],
  "sql_answer_query": "SELECT AVG(value) FROM unified_data WHERE metric = 'net_flow' AND sname = 'Tonghuashun' AND code = '300033.SZ' AND tdate BETWEEN '2023-12-01' AND '2023-12-03'",
  "sql_evidence_query": "SELECT * FROM unified_data WHERE metric = 'net_flow' AND sname = 'Tonghuashun' AND code = '300033.SZ' AND tdate BETWEEN '2023-12-01' AND '2023-12-03'"
}}
""",
    
    # 金融领域 - 中等难度（带条件聚合）
    "financial_structured_medium_template_en": """
### Task: Generate Conditional Financial Aggregation Query with SQL
Generate a question that requires an aggregation with filtering conditions or grouping and corresponding SQL queries.

### Available Information (values in RMB million)
{session_context}

### Rules
1. **Question Generation Rules**:
   - Use professional financial terminology
   - Include specific filtering conditions (e.g., value ranges, metrics)
   - Scope can include multiple entities or longer time range (4-7 days)
   - Frame questions in a business context
   - Question MUST clearly specify the stock name and code
  
2. **SQL Generation Rules**:
   - Use EXACT variable names from the session context
   - Ensure SQLite compatibility(sqlite3)
   - SQL_ANSWER query must directly yield the answer to the question
   - SQL_EVIDENCE query must retrieve all rows used to yield the answer
   - Both queries must be syntactically correct and executable

3. **Data-Specific Rules**:
   - `code`: Stock code
   - `sname`: Stock name
   - `tdate`: Transaction date
   - `value`: Numeric value
   - `metric`: Financial metric

4. Output JSON only:
{{
  "question": "...?",
  "answer": <float | int>,
  "evidence": [
    ["code", "sname", "YYYY-MM-DD", <value>, <metric>],
    ...
  ],
  "sql_answer_query": "SELECT ...",
  "sql_evidence_query": "SELECT ..."
}}

### Example
{{
  "question": "What was the total net capital inflow for Alibaba Group (BABA.NYSE) in the first week of December 2023 for days with net flow above 200 million RMB?",
  "answer": 1250.0,
  "evidence": [
    ["BABA.NYSE", "Alibaba Group", "2023-12-01", 350.0, "net_flow"],
    ["BABA.NYSE", "Alibaba Group", "2023-12-03", 420.0, "net_flow"],
    ["BABA.NYSE", "Alibaba Group", "2023-12-05", 480.0, "net_flow"],
    ["BABA.NYSE", "Alibaba Group", "2023-12-02", 180.0, "net_flow"],
    ["BABA.NYSE", "Alibaba Group", "2023-12-04", 150.0, "net_flow"],
    ["BABA.NYSE", "Alibaba Group", "2023-12-06", 120.0, "net_flow"],
    ["BABA.NYSE", "Alibaba Group", "2023-12-07", 90.0, "net_flow"]
  ],
  "sql_answer_query": "SELECT SUM(value) FROM unified_data WHERE metric = 'net_flow' AND sname = 'Alibaba Group' AND code = 'BABA.NYSE' AND tdate BETWEEN '2023-12-01' AND '2023-12-07' AND value > 200",
  "sql_evidence_query": "SELECT * FROM unified_data WHERE metric = 'net_flow' AND sname = 'Alibaba Group' AND code = 'BABA.NYSE' AND tdate BETWEEN '2023-12-01' AND '2023-12-07'"
}}
""",
    
    # 金融领域 - 困难难度（复杂聚合）
    "financial_structured_hard_template_en": """
### Task: Generate Complex Financial Analysis Query with SQL
Generate a financial question requiring multiple aggregation steps and corresponding SQL queries.

### Available Information (values in RMB million)
{session_context}

### Rules
1. **Question Generation Rules**:
   - Use professional financial terminology
   - Include multiple metrics and time-based comparisons
   - Scope must include multiple entities and/or longer time ranges (2+ weeks)
   - Frame questions in a business analysis context requiring complex calculations
   - Question MUST clearly specify the stock names and codes
   - Include at least two different aggregation functions or subqueries

2. **SQL Generation Rules**:
   - Use EXACT variable names from the session context
   - Ensure SQLite compatibility(sqlite3)
   - SQL_ANSWER query must directly yield the answer to the question
   - SQL_EVIDENCE query must retrieve all rows used to yield the answer
   - Both queries must be syntactically correct and executable

3. **Data-Specific Rules**:
   - `code`: Stock code
   - `sname`: Stock name
   - `tdate`: Transaction date
   - `value`: Numeric value
   - `metric`: Financial metric

4. Output JSON only:
{{
  "question": "...?",
  "answer": <float | int>,
  "evidence": [
    ["code", "sname", "YYYY-MM-DD", <value>, <metric>],
    ...
  ],
  "sql_answer_query": "SELECT ...",
  "sql_evidence_query": "SELECT ..."
}}

### Example
{{
  "question": "What is the percentage difference between Tencent Holdings (700.HK)'s average daily trading volume and PetroChina (601857.SH)'s average daily trading volume during the first two weeks of December 2023?",
  "answer": 35.8,
  "evidence": [
    ["700.HK", "Tencent Holdings", "2023-12-01", 1250.0, "volume"],
    ["700.HK", "Tencent Holdings", "2023-12-04", 1320.0, "volume"],
    ["700.HK", "Tencent Holdings", "2023-12-07", 1180.0, "volume"],
    ["700.HK", "Tencent Holdings", "2023-12-11", 1420.0, "volume"],
    ["700.HK", "Tencent Holdings", "2023-12-14", 1380.0, "volume"],
    ["601857.SH", "PetroChina", "2023-12-01", 850.0, "volume"],
    ["601857.SH", "PetroChina", "2023-12-04", 920.0, "volume"],
    ["601857.SH", "PetroChina", "2023-12-07", 880.0, "volume"],
    ["601857.SH", "PetroChina", "2023-12-11", 1050.0, "volume"],
    ["601857.SH", "PetroChina", "2023-12-14", 980.0, "volume"]
  ],
  "sql_answer_query": "SELECT ((SELECT AVG(value) FROM unified_data WHERE metric = 'volume' AND sname = 'Tencent Holdings' AND code = '700.HK' AND tdate BETWEEN '2023-12-01' AND '2023-12-14') - (SELECT AVG(value) FROM unified_data WHERE metric = 'volume' AND sname = 'PetroChina' AND code = '601857.SH' AND tdate BETWEEN '2023-12-01' AND '2023-12-14') ) / (SELECT AVG(value) FROM unified_data WHERE metric = 'volume' AND sname = 'PetroChina' AND code = '601857.SH' AND tdate BETWEEN '2023-12-01' AND '2023-12-14') * 100",
  "sql_evidence_query": "SELECT * FROM unified_data WHERE metric = 'volume' AND ((sname = 'Tencent Holdings' AND code = '700.HK') OR (sname = 'PetroChina' AND code = '601857.SH')) AND tdate BETWEEN '2023-12-01' AND '2023-12-14'"
}}
""",
    
    # 医疗领域 - 简单难度（单次聚合）
    "medical_structured_easy_template_en": """
    ### Task: Generate Simple Clinical Aggregation Question with SQL
    Generate a question that requires a single aggregation (AVG, MAX, MIN, SUM) 
    over a small set of clinical values with specific time references and corresponding SQL queries.
    
    ### Available Information
    {session_context}
    
    ### Rules
    1. **Question Generation Rules**:
       - Use scientifically precise and clinically relevant terminology. Example: "serum glucose level" instead of "Glucose"
       - Include specific clinical parameters and time ranges (within one day or a few hours)
       - Frame questions in a clinical context
       - Question MUST clearly specify the patient ID
       - Limited to a single parameter type
    
    2. **SQL Generation Rules**:
       - Use EXACT variable names from the session context
       - Ensure SQLite compatibility(sqlite3)
       - SQL_ANSWER query must calculate the answer to the question
       - SQL_EVIDENCE query must retrieve all rows used to yield the answer
       - Both queries must be syntactically correct and executable
    
    3. **Data-Specific Rules**:
       - `PatientID`: Patient identifier (e.g., "OPO1_P1000")
       - `time_event`: Measurement time (format: "YYYY-MM-DD HH:MM:SS")
       - `variable_name`: Clinical parameter name (e.g., "O2SAT", "Glucose")
       - `value`: Numeric measurement value
       - `table_type`: Data source (e.g., "ABGEvents", "ChemistryEvents")
    
    4. Output JSON only:
    {{
      "question": "...?",
      "answer": <float | int>,
      "evidence": [
        ["PatientID", "time_stamp", "variable_name", <value>, "table_type"],
        ...
      ],
      "sql_answer_query": "SELECT ...",
      "sql_evidence_query": "SELECT ..."
    }}
    
    ### Example
    {{
      "question": "What was patient OPO1_P1000's average arterial oxygen saturation (O2SAT) during SIMV ventilation between 08:00 and 16:00 on June 21, 2023?",
      "answer": 95.7,
      "evidence": [
        ["OPO1_P1000", "2023-06-21 08:00:00", "SIMV-O2SAT", 96.0, "ABGEvents"],
        ["OPO1_P1000", "2023-06-21 12:00:00", "SIMV-O2SAT", 95.0, "ABGEvents"],
        ["OPO1_P1000", "2023-06-21 16:00:00", "SIMV-O2SAT", 96.0, "ABGEvents"]
      ],
      "sql_answer_query": "SELECT AVG(value) FROM unified_data WHERE PatientID = 'OPO1_P1000' AND variable_name = 'SIMV-O2SAT' AND table_type = 'ABGEvents' AND time_event BETWEEN '2023-06-21 08:00:00' AND '2023-06-21 16:00:00'",
      "sql_evidence_query": "SELECT * FROM unified_data WHERE PatientID = 'OPO1_P1000' AND variable_name = 'SIMV-O2SAT' AND table_type = 'ABGEvents' AND time_event BETWEEN '2023-06-21 08:00:00' AND '2023-06-21 16:00:00'"
    }}
    """,
    
    # 医疗领域 - 中等难度（带条件聚合）
    "medical_structured_medium_template_en": """
### Task: Generate Conditional Clinical Question with SQL
Generate a question that requires an aggregation with filtering conditions and specific time references and corresponding SQL queries.

### Available Information
{session_context}

### Rules
1. **Question Generation Rules**:
   - Use scientifically precise and clinically relevant terminology. Example: "serum glucose level" instead of "Glucose"
   - Question must involve one aggregation function, and include at least one filtering condition with explicit time references (2-5 days)
   - Frame questions in a clinical context with conditions
   - If using clinical terms (e.g., "postoperative"), include specific dates in parentheses
   - Question MUST clearly specify the patient ID
   - May include multiple related parameters

2. **SQL Generation Rules**:
   - Use EXACT variable names from the session context
   - Ensure SQLite compatibility(sqlite3)
   - SQL_ANSWER query must calculate the answer to the question
   - SQL_EVIDENCE query must retrieve all rows to yield the answer
   - Both queries must be syntactically correct and executable

3. **Data-Specific Rules**:
   - `PatientID`: Patient identifier
   - `time_event`: Measurement time (format: "YYYY-MM-DD HH:MM:SS")
   - `variable_name`: Clinical parameter name
   - `value`: Numeric measurement value
   - `table_type`: Data source (e.g., "ABGEvents", "ChemistryEvents")

4. Output JSON only:
{{
  "question": "...?",
  "answer": <float | int>,
  "evidence": [
    ["PatientID", "time_event", "variable_name", <value>, "table_type"],
    ...
  ],
  "sql_answer_query": "SELECT ...",
  "sql_evidence_query": "SELECT ..."
}}

### Example
{{
  "question": "What was the minimum arterial pH value observed for patient OPO1_P10004 during SIMV ventilation from 23:10 on April 12, 2036 to 09:05 on April 15, 2036, when the pH was below 7.35?",
  "answer": 7.11,
  "evidence": [
    ["OPO1_P10004", "2036-04-12 23:10:00", "SIMV-PH", 7.11, "ABGEvents"],
    ["OPO1_P10004", "2036-04-13 02:03:00", "SIMV-PH", 7.37, "ABGEvents"],
    ["OPO1_P10004", "2036-04-14 07:26:00", "SIMV-PH", 7.34, "ABGEvents"],
    ["OPO1_P10004", "2036-04-15 05:52:00", "SIMV-PH", 7.35, "ABGEvents"],
    ["OPO1_P10004", "2036-04-15 09:05:00", "SIMV-PH", 7.33, "ABGEvents"]
  ],
  "sql_answer_query": "SELECT MIN(value) FROM unified_data WHERE PatientID = 'OPO1_P10004' AND variable_name = 'SIMV-PH' AND table_type = 'ABGEvents' AND time_event BETWEEN '2036-04-12 23:10:00' AND '2036-04-15 09:05:00' AND value < 7.35",
  "sql_evidence_query": "SELECT * FROM unified_data WHERE PatientID = 'OPO1_P10004' AND variable_name = 'SIMV-PH' AND table_type = 'ABGEvents' AND time_event BETWEEN '2036-04-12 23:10:00' AND '2036-04-15 09:05:00'"
}}
""",
    
    # 医疗领域 - 困难难度（复杂聚合）
    "medical_structured_hard_template_en": """
### Task: Generate Complex Clinical Analysis Question with SQL
Generate a clinical question requiring multiple aggregation steps and corresponding SQL queries.

### Available Information
{session_context}

### Rules
1. **Question Generation Rules**:
   - Use scientifically precise and clinically relevant terminology. Example: "serum glucose level" instead of "Glucose"
   - Question must involve at least two aggregation functions or complex calculations
   - Include time-based comparisons with explicit date ranges (7+ days)
   - Frame questions in a clinical analysis context requiring advanced statistical understanding
   - Question MUST clearly specify the patient ID
   - Must include multiple parameters or complex relationships between parameters

2. **SQL Generation Rules**:
   - Use EXACT variable names from the session context
   - Ensure SQLite compatibility(sqlite3)
   - SQL_ANSWER query must calculate the answer to the question
   - SQL_EVIDENCE query must retrieve all rows to yield the answer
   - Both queries must be syntactically correct and executable

3. **Data-Specific Rules**:
   - `PatientID`: Patient identifier
   - `time_event`: Measurement time (format: "YYYY-MM-DD HH:MM:SS")
   - `variable_name`: Clinical parameter name
   - `value`: Numeric measurement value
   - `table_type`: Data source (e.g., "ABGEvents", "ChemistryEvents")

4. Output JSON only:
{{
  "question": "...?",
  "answer": <float | int>,
  "evidence": [
    ["PatientID", "time_stamp", "variable_name", <value>, "table_type"],
    ...
  ],
  "sql_answer_query": "SELECT ...",
  "sql_evidence_query": "SELECT ..."
}}

### Example
{{
  "question": "For patient OPO1_P1000, calculate the percentage change in the ratio of average arterial oxygen saturation (O2SAT) to average partial pressure of carbon dioxide (PCO2) between the first week of June 2023 (June 1-7) and the second week (June 8-14)?",
  "answer": -8.2,
  "evidence": [
    ["OPO1_P1000", "2023-06-01 08:00:00", "SIMV-O2SAT", 98.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-01 08:00:00", "SIMV-PCO2", 38.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-03 12:00:00", "SIMV-O2SAT", 96.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-03 12:00:00", "SIMV-PCO2", 40.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-07 16:00:00", "SIMV-O2SAT", 97.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-07 16:00:00", "SIMV-PCO2", 39.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-08 08:00:00", "SIMV-O2SAT", 93.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-08 08:00:00", "SIMV-PCO2", 42.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-10 12:00:00", "SIMV-O2SAT", 94.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-10 12:00:00", "SIMV-PCO2", 44.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-14 16:00:00", "SIMV-O2SAT", 92.0, "ABGEvents"],
    ["OPO1_P1000", "2023-06-14 16:00:00", "SIMV-PCO2", 45.0, "ABGEvents"]
  ],
  "sql_answer_query": "SELECT (((SELECT AVG(o2.value) FROM unified_data o2 WHERE o2.PatientID = 'OPO1_P1000' AND o2.variable_name = 'SIMV-O2SAT' AND o2.table_type = 'ABGEvents' AND o2.time_event BETWEEN '2023-06-08 00:00:00' AND '2023-06-14 23:59:59') / (SELECT AVG(pco2.value) FROM unified_data pco2 WHERE pco2.PatientID = 'OPO1_P1000' AND pco2.variable_name = 'SIMV-PCO2' AND pco2.table_type = 'ABGEvents' AND pco2.time_event BETWEEN '2023-06-08 00:00:00' AND '2023-06-14 23:59:59')) - ((SELECT AVG(o2.value) FROM unified_data o2 WHERE o2.PatientID = 'OPO1_P1000' AND o2.variable_name = 'SIMV-O2SAT' AND o2.table_type = 'ABGEvents' AND o2.time_event BETWEEN '2023-06-01 00:00:00' AND '2023-06-07 23:59:59') / (SELECT AVG(pco2.value) FROM unified_data pco2 WHERE pco2.PatientID = 'OPO1_P1000' AND pco2.variable_name = 'SIMV-PCO2' AND pco2.table_type = 'ABGEvents' AND pco2.time_event BETWEEN '2023-06-01 00:00:00' AND '2023-06-07 23:59:59'))) / ((SELECT AVG(o2.value) FROM unified_data o2 WHERE o2.PatientID = 'OPO1_P1000' AND o2.variable_name = 'SIMV-O2SAT' AND o2.table_type = 'ABGEvents' AND o2.time_event BETWEEN '2023-06-01 00:00:00' AND '2023-06-07 23:59:59') / (SELECT AVG(pco2.value) FROM unified_data pco2 WHERE pco2.PatientID = 'OPO1_P1000' AND pco2.variable_name = 'SIMV-PCO2' AND pco2.table_type = 'ABGEvents' AND pco2.time_event BETWEEN '2023-06-01 00:00:00' AND '2023-06-07 23:59:59')) * 100",
  "sql_evidence_query": "SELECT * FROM unified_data WHERE PatientID = 'OPO1_P1000' AND (variable_name = 'SIMV-O2SAT' OR variable_name = 'SIMV-PCO2') AND table_type = 'ABGEvents' AND time_event BETWEEN '2023-06-01 00:00:00' AND '2023-06-14 23:59:59'"
}}
""",
})

PERSONA = ({
    "financial": {
        "user": "You are a prudent individual investor reviewing financial market data and corporate reports. You focus on key financial metrics, market trends, and risk indicators to make investment decisions. You ask specific, targeted questions about: financial performance metrics, market trends, risk indicators, comparative analysis between different time periods, and data-driven investment recommendations. Your communication style is pragmatic and inquisitive, emphasizing thorough research and professional analysis before making investment decisions.",
        "assistant": "You are a professional financial analyst with expertise in market analysis and investment advisory. You can: 1) Accurately interpret financial statements and market indicators, 2) Identify market trends and potential risks, 3) Provide clear data summaries and trend analysis, 4) Offer evidence-based investment recommendations, 5) Explain complex financial metrics in accessible terms. You use precise yet understandable financial terminology, avoiding both oversimplification and unnecessary complexity. Responses are structured: key findings first, then detailed market analysis, followed by actionable recommendations. For significant market movements or anomalies, emphasize potential risks and opportunities."
    },
    "medical": {
        "user": "You are an ICU attending physician reviewing a critical patient's medical records. You focus on trends and abnormalities in key vital signs and lab indicators to make treatment decisions. You ask specific, targeted questions to obtain crucial information such as: key metrics at specific timepoints, trend changes, clinical significance of abnormalities, correlations between multiple indicators, and data-driven treatment recommendations.",
        "assistant": "You are a professional medical data analyst with clinical background and data analysis expertise. You can: 1) Accurately interpret lab results and vital signs data, 2) Identify trends and abnormalities in key metrics, 3) Provide clear data summaries and visual descriptions, 4) Offer evidence-based treatment recommendations, 5) Explain clinical significance of complex metrics. You use precise yet accessible medical terminology, avoiding oversimplification or unnecessary complexity. Responses are structured: key findings summary first, then detailed data explanation, followed by clinical recommendations. For abnormalities, emphasize clinical significance and potential risks."
    }
})

SESSION_SIMULATOR_PROMPT: Dict[str, str] = ({
})
SESSION_SIMULATOR_PROMPT["financial"] = ({
    "user": """
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
"I've been looking at Tonghuashun (300033.SZ)'s capital flow data for December 2023. For example, on Dec 1st, there was an inflow of RMB 279 million, followed by RMB 570 million on Dec 4th, and RMB 456 million on Dec 8th. However, the trend shifted, with an outflow of RMB 148.58 million on Dec 13th, and RMB 212.77 million on Dec 14th. What's your analysis of these fluctuations and their potential impact?"

EVIDENCES_USED_IN_THIS_TURN:
- ('300033.SZ', 'Tonghuashun', '2023-12-01', 279.0, 'net_flow')
- ('300033.SZ', 'Tonghuashun', '2023-12-04', 570.0, 'net_flow')
- ('300033.SZ', 'Tonghuashun', '2023-12-08', 456.0, 'net_flow')
- ('300033.SZ', 'Tonghuashun', '2023-12-13', -148.58, 'net_flow')
- ('300033.SZ', 'Tonghuashun', '2023-12-14', -212.77, 'net_flow')

Example for Option 2 (query for specific time period):
"Can you give me the daily net inflow for Tonghuashun (300033.SZ) during the first ten days of December 2023?"

EVIDENCES_USED_IN_THIS_TURN:
none
for Option 2 (This section would be empty as the user is *requesting* new data, not referencing existing data from the list):
""",

    "assistant": """
{persona}

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
When user ask: "Can you give me the daily net inflow for Tonghuashun (300033.SZ) during the first ten days of December 2023?"

You should response(remember that all the data are from the list "Remaining **Un-discussed Financial Data**", if there is no data in the list you needed, you should response "Sorry, I don't have the data for that time period."):
Sure, here are the daily net inflow for Tonghuashun (300033.SZ) during the first three days of December 2023:
- Dec 1: RMB 279 million
- Dec 2: RMB 570 million
- Dec 3: RMB 456 million

EVIDENCES_USED_IN_THIS_TURN:
- ('300033.SZ', 'Tonghuashun', '2023-12-01', 279.0, 'net_flow')
- ('300033.SZ', 'Tonghuashun', '2023-12-02', 570.0, 'net_flow')
- ('300033.SZ', 'Tonghuashun', '2023-12-03', 456.0, 'net_flow')
"""
})

SESSION_SIMULATOR_PROMPT["medical"] = {
    "user": """
{persona}

Your overarching goal is to ensure a **complete and thorough discussion** of **ALL** the medical data that still needs to be covered.

Remaining **Un-discussed Medical Data** for this session:
{evidences}

---

Decision-making process
1. Examine the `Remaining Un-discussed Medical Data` list provided.
2. **Query for Specific Patient Data**: Formulate a specific question asking for data about a particular medical parameter or time period. 
3. Tone: professional, concerned about patient care.
4. **IMPORTANT**: Only reference data points that are EXPLICITLY listed in the `Remaining Un-discussed Medical Data`. DO NOT invent or assume the existence of any data not provided.
5. If you're asking a question without referencing specific data points, your EVIDENCES_USED_IN_THIS_TURN block MUST be empty.
6. **After your message**, list every **exact original tuple evidence** you just explicitly implicated. Each evidence MUST be on a new line, starting with '- ', and contain ONLY ONE complete tuple. This `EVIDENCES_USED_IN_THIS_TURN:` block is NOT part of chat history.

Last Assistant Response:
{last_turn_content}

### Example 
Example for Option 1 (present data + ask for analysis):
"I'm reviewing Patient OPO1_P100082's lab results. I see their Blood culture was negative at 17:29 on June 6, 2036, and remained negative at 17:38 the same day. What's your interpretation of these results and what should we monitor next?"

EVIDENCES_USED_IN_THIS_TURN:
- ('OPO1_P100082', '2036-06-06 17:29:00', 'CultureEvents', 'Blood_culture', 0.0)
- ('OPO1_P100082', '2036-06-06 17:38:00', 'CultureEvents', 'Blood_culture', 0.0)

Example for Option 2 (query for specific patient data):
"Can you tell me all the lab results for Patient OPO1_P100082 from June 6, 2036?"

EVIDENCES_USED_IN_THIS_TURN:
none
""",

    "assistant": """
{persona}

Remaining **Un-discussed Medical Data**: 
{evidences}

---

Decision-making process 
1. **If the user supplied data** → analyze directly with clinical insight. 
2. **If the user requested data** → retrieve and present the relevant **tuple evidences** from the list. 
3. **CRITICAL RULE**: ONLY use data points that are EXPLICITLY listed in the `Remaining Un-discussed Medical Data`. NEVER invent, assume, or hallucinate data that is not in this list.
4. If a user asks for data that doesn't exist in the list, clearly state that you don't have that information.
5. When listing evidence used, ONLY include tuples that are EXACTLY as they appear in the `Remaining Un-discussed Medical Data` list.
6. **After answering**, proactively surface any **still-un-discussed tuple evidences** when natural, but only if they exist in the list.
7. Keep tone professional and clinically relevant.

Last Turn: 
{last_turn_content}

Example:
When user asks: "Can you tell me all the lab results for Patient OPO1_P100082 from June 6, 2036?"

You should respond (remember that all the data are from the list "Remaining **Un-discussed Medical Data**", if there is no data in the list you needed, you should respond "I don't have any lab results for that specific date."):

Here are the lab results for Patient OPO1_P100082 from June 6, 2036:
- Blood culture at 17:29: Negative
- Blood culture at 17:38: Negative

These negative blood cultures suggest no bacterial growth was detected in the samples. This is generally a good sign indicating absence of bacteremia, though it's important to correlate with other clinical findings and the patient's overall condition.

EVIDENCES_USED_IN_THIS_TURN:
- ('OPO1_P100082', '2036-06-06 17:29:00', 'CultureEvents', 'Blood_culture', 0.0)
- ('OPO1_P100082', '2036-06-06 17:38:00', 'CultureEvents', 'Blood_culture', 0.0)
"""
}