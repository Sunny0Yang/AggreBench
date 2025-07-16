from typing import Dict

QA_GENERATION_PROMPTS: Dict[str, str] =({
    "conversational_easy_template_en": """
Please generate a *simple, direct* aggregative query question with practical business/workplace relevance based on the provided sessions.
This question should require a single, straightforward aggregation (e.g., counting direct occurrences, summing a clear metric).

{session_context}

Generation Requirements:
1. **Core Operations**: MUST use counting, summing, or deduplication (single operation).
2. **Scope**: MUST span at least {session_threshold} sessions.
3. **Simplicity**: Avoid complex filtering or multi-step reasoning. Focus on direct observations.
4. **Real-World Significance**: Questions should reveal:
   - Emerging problems/trends (e.g., recurring errors, sentiment shifts)
   - Resource needs (e.g., frequently requested features/tools)
   - Customer impact (e.g., affected user groups/regions)
   - Operational patterns (e.g., common request types, bottlenecks)
5. **Output Format**: Strict JSON only:
{{
    "question": "Question text",
    "answer": "The answer is: [numeric value or concise response]",  # MUST start with "The answer is:"
    "evidence": ["Evidence 1", "Evidence 2", ...]  # Exactly {min_evidences}-{max_evidences} entries
}}
6. **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include evidence that is absolutely necessary to derive the answer.**
    - If there aren't enough direct evidences to meet 'min_evidences', *do not generate the question*.
    - References MUST use "D{{session_id}}:{{turn_id}}" format.
    - Quotes MUST be verbatim and directly support the answer.
    - Each evidence entry MUST contribute to demonstrating the root cause or solution context.
    - Must contain at least {min_evidences} but no more than {max_evidences} distinct evidences.
7. **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For list answers: "The answer is: [item1], [item2], ..."
    - For complex answers: "The answer is: [concise summary]"
    - Avoid explanations or justifications in the answer field

Example Output:
{{
    "question": "How many times was 'payment failure' mentioned across these sessions?",
    "answer": "The answer is: 3",
    "evidence": [
        "D2:8: 'Payment failed for order EU-1092'",
        "D3:1: 'client unable to process payment'",
        "D5:6: 'Payment gateway error'"
    ]
}}
""",

   "conversational_medium_template_en": """
Please generate an aggregative query question of *medium difficulty* with practical business/workplace relevance based on the provided sessions.
This question should involve either:
- Multiple filtering conditions.
- Aggregation across specific time periods.
- Simple comparison between two aggregated values.

{session_context}

Generation Requirements:
1. **Core Operations**: MUST use counting, summing, or deduplication.
2. **Scope**: MUST span at least {session_threshold} sessions.
3. **Medium Complexity**: Incorporate 1-2 filtering conditions (e.g., 'where X and Y'), or focus on time-based aggregation (e.g., daily counts), or compare two simple aggregated results.
4. **Real-World Significance**: Questions should reveal:
   - Emerging problems/trends (e.g., recurring errors, sentiment shifts)
   - Resource needs (e.g., frequently requested features/tools)
   - Customer impact (e.g., affected user groups/regions)
   - Operational patterns (e.g., common request types, bottlenecks)
5. **Output Format**: Strict JSON only:
{{
    "question": "Question text",
    "answer": "The answer is: [numeric value or concise response]",
    "evidence": ["Evidence 1", "Evidence 2", ...]  # Exactly {min_evidences}-{max_evidences} entries
}}
6. **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include evidence that is absolutely necessary to derive the answer.**
    - If there aren't enough direct evidences to meet 'min_evidences', *do not generate the question*.
    - References MUST use "D{{session_id}}:{{turn_id}}" format.
    - Quotes MUST be verbatim and directly support the answer.
    - Each evidence entry MUST contribute to demonstrating the root cause or solution context.
    - Must contain at least {min_evidences} but no more than {max_evidences} distinct evidences.
7. **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For list answers: "The answer is: [item1], [item2], ..."
    - For complex answers: "The answer is: [concise summary]"
    - Avoid explanations or justifications in the answer field

Example Output:
{{
    "question": "In sessions from January, how many distinct users reported 'login' issues?",
    "answer": "The answer is: 2",
    "evidence": [
        "D1:2: 'User A reported login problem on Jan 5th'",
        "D4:7: 'Login issue persisted for User B, Jan 12th'"
    ]
}}
""",

    "conversational_hard_template_en": """
Please generate a *complex, multi-step* aggregative query question with high practical business/workplace relevance based on the provided sessions.
This question should require:
- Multiple aggregations (e.g., average of sums, count of distinct items meeting specific criteria, or multi-level grouping).
- Or, complex temporal reasoning (e.g., trend over time, comparison of periods, specific sequence of events).
- Or, strong implicit filtering / requires deeper contextual understanding.

{session_context}

Generation Requirements:
1. **Core Operations**: MUST use multiple counting, summing, or deduplication operations, or a combination.
2. **Scope**: MUST span at least {session_threshold} sessions, potentially requiring integration of subtle information.
3. **High Complexity**:
    - Involve nested aggregations (e.g., "sum of daily averages").
    - Or, require complex temporal reasoning (e.g., "trend of issues over weekly periods").
    - Or, infer complex relationships or conditions from the context.
4. **Real-World Significance**: Questions should reveal:
   - Emerging problems/trends (e.g., recurring errors, sentiment shifts)
   - Resource needs (e.g., frequently requested features/tools)
   - Customer impact (e.g., affected user groups/regions)
   - Operational patterns (e.g., common request types, bottlenecks)
5. **Output Format**: Strict JSON only:
{{
    "question": "Question text",
    "answer": "The answer is: [numeric value or concise response]",
    "evidence": ["Evidence 1", "Evidence 2", ...]  # Exactly {min_evidences}-{max_evidences} entries
}}
6. **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include evidence that is absolutely necessary to derive the answer.**
    - If there aren't enough direct evidences to meet 'min_evidences', *do not generate the question*.
    - References MUST use "D{{session_id}}:{{turn_id}}" format.
    - Quotes MUST be verbatim and directly support the answer.
    - Each evidence entry MUST contribute to demonstrating the root cause or solution context.
    - Must contain at least {min_evidences} but no more than {max_evidences} distinct evidences.
7. **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For list answers: "The answer is: [item1], [item2], ..."
    - For complex answers: "The answer is: [concise summary]"
    - Avoid explanations or justifications in the answer field

Example Output:
{{
    "question": "What was the average daily number of critical bugs reported across all weeks, only considering weekdays?",
    "answer": "The answer is: 1.5",
    "evidence": [
        "D1:5: 'Critical bug reported, Mon Jan 3'",
        "D3:2: 'Another critical bug, Tue Jan 4'",
        "D6:1: 'Critical bug found on Wed Jan 12'",
        "D8:4: 'New critical bug today, Thu Jan 13'"
    ]
}}
""",

    "structured_easy_template_en": """
Please generate a *simple, direct* aggregative query question based on the provided structured table data.
This question should require a single, straightforward aggregation (e.g., COUNT, SUM, AVG, MAX, MIN) focusing on one specific column.

{session_context}

Generation Requirements (Easy Level):
1.  **Core Operation**: MUST use a single aggregation function (COUNT, SUM, AVG, MAX, MIN).
2.  **Scope**: Focus on one specific column, no complex filtering or joins required.
3.  **Real-World Significance**: Questions should reveal direct facts or simple totals/averages from the data.
4.  **Output Format**: Strict JSON only:
{{
    "question": "Question text",
    "answer": "The answer is: [numeric value or concise response]", # MUST start with "The answer is:"
    "evidence": ["<Identifier Column Name>: <Identifier Value>; <Target Column Name>: <Target Value>"] # Exactly 1 evidence entry.
}}
5.  **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include the single, most direct evidence entry that is absolutely necessary to derive the answer.**
    - References MUST use "<Identifier Column Name>: <Identifier Value>; <Target Column Name>: <Target Value>" format.
    - The evidence provided must directly contain the value or data points used in the aggregation.
    - Must contain exactly 1 distinct evidence entry.
6.  **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For numerical answers, please use base units (e.g., if the answer is '5.20 亿元', convert it to '520000000 元'). If the answer is a percentage, convert it to a decimal (e.g., 25%% should be 0.25).
    - Avoid explanations or justifications in the answer field.

Example:
{{
    "question": "What is the maximum capital outflow for 万科A on any given date in the provided data?",
    "answer": "The answer is: 520000000 元",
    "evidence": ["股票简称: 万科A; 资金流出[20231212]: 5.25亿元"]
}}
""",
    
"structured_medium_template_en": """
Please generate an aggregative query question of *medium difficulty* based on the provided structured table data.
This question should involve either:
- A two-step aggregation (e.g., AVG requires SUM and COUNT implicitly).
- Aggregation on one column with simple filtering conditions.
- Simple comparison between two aggregated values from the same table.

{session_context}

Generation Requirements (Medium Level):
1.  **Core Operation**: MUST use a two-step aggregation (e.g., AVG implies SUM and COUNT) OR a single aggregation with simple filtering (e.g., COUNT WHERE X > Y).
2.  **Scope**: Involve one column with 1-2 simple filtering conditions, or direct comparison of two aggregated results from the same table. No complex joins across multiple implicit "tables" unless they are presented as clearly separate blocks within the session context.
3.  **Real-World Significance**: Questions should reveal insights that require a bit more processing than direct lookup, but are still straightforward.
4.  **Output Format**: Strict JSON only:
{{
    "question": "Question text",
    "answer": "The answer is: [numeric value or concise response]",
    "evidence": ["<Identifier Column Name>: <Identifier Value>; <Column_Name_1>: <value_1>; <Column_Name_2>: <value_2>", ...] # 2-3 evidence entries.
}}
5.  **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include evidence that is absolutely necessary to derive the answer.**
    - If there aren't enough direct evidences to meet 'min_evidences' (2), *do not generate the question*.
    - References MUST use "<Identifier Column Name>: <Identifier Value>; <Column_Name_1>: <value_1>; <Column_Name_2>: <value_2>" format. If multiple column/value pairs are from the same identifier, separate them with a semicolon ';'.
    - Quotes MUST be verbatim and directly support the answer's calculation or claim.
    - Each evidence entry MUST contribute directly to the calculation or understanding required to answer.
    - Must contain 2-3 distinct evidence entries.
6.  **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For numerical answers, please use base units (e.g., if the answer is '5.20 亿元', convert it to '520000000 元'). If the answer is a percentage, convert it to a decimal (e.g., 25%% should be 0.25).
    - Avoid explanations or justifications in the answer field.

Example:
{{
    "question": "What is the average daily capital inflow for 同花顺 (300033.SZ) in the first week of December 2023 (Dec 1st to Dec 7th)?",
    "answer": "The answer is: 263435420.0",
    "evidence": [
        "股票简称: 同花顺; 资金流向[20231201]: 2.79亿元",
        "股票简称: 同花顺; 资金流向[20231204]: 5.70亿元",
        "股票简称: 同花顺; 资金流向[20231205]: 4814.18万元"
    ]
}}
""",
    
    "structured_hard_template_en": """
Please generate a *complex, multi-step* aggregative query question based on the provided structured table data.
This question should require:
- Multiple nested aggregations or complex calculations across different columns.
- Sophisticated filtering, potentially involving multiple conditions or implied relationships.
- Time-based analysis or comparisons (e.g., year-over-year growth, trends over periods).
- Potentially combine information from multiple conceptual "tables" if the context implies them (e.g., different sections of the data representing different entities or timeframes that need to be linked).

{session_context}

Generation Requirements (Hard Level):
1.  **Core Operation**: MUST involve multi-step aggregation (e.g., nested calculations like "average of daily sums", or calculations across multiple related columns).
2.  **Scope**: Combine multiple columns, potentially requiring implicit joins or complex logical deductions.
3.  **Time Dimension**: MUST include time-based filtering, comparisons, or trend analysis.
4.  **Real-World Significance**: Questions should reveal complex trends, anomalies, or strategic insights that require advanced data manipulation.
5.  **Output Format**: Strict JSON only:
{{
    "question": "Question text",
    "answer": "The answer is: [numeric value or concise response]",
    "evidence": ["<Identifier>: <Column_Name>: <value>; <Another_Column_Name>: <value>", ...] # 4-5 evidence entries.
}}
6.  **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include evidence that is absolutely necessary to derive the answer.**
    - If there aren't enough direct evidences to meet 'min_evidences' (4), *do not generate the question*.
    - References MUST use "<Identifier>: <Column_Name>: <value>" format. If multiple column/value pairs are from the same identifier, separate them with a semicolon ';'.
    - Quotes MUST be verbatim and directly support the answer's calculation or claim.
    - Each evidence entry MUST contribute directly to the complex calculation or understanding required to answer.
    - Must contain 4-5 distinct evidence entries.
7.  **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For numerical answers, please use base units (e.g., if the answer is '5.20 亿元', convert it to '520000000 元'). If the answer is a percentage, convert it to a decimal (e.g., 25%% should be 0.25).
    - Avoid explanations or justifications in the answer field.

Example:
{{
    "question": "Calculate the percentage change in net capital flow (inflow minus outflow) from December 1–7 to December 25–31, 2023 and report the single rounded-to-two-decimals figure?",
    "answer": "The answer is: 0.36",
    "evidence": [
        "股票简称: 同花顺; 资金流向[20231201]: 2.79亿元",
        "股票简称: 同花顺; 资金流出[20231201]: 5.04亿元",
        "股票简称: 同花顺; 资金流向[20231204]: 5.70亿元",
        "股票简称: 同花顺; 资金流出[20231204]: 6.80亿元",
        "股票简称: 同花顺; 资金流向[20231222]: -112451513.00元",
        "股票简称: 同花顺; 资金流出[20231222]: 4.29亿元",
        "股票简称: 同花顺; 资金流向[20231225]: -21258570.00元",
        "股票简称: 同花顺; 资金流出[20231225]: 2.32亿元",
        "股票简称: 同花顺; 资金流向[20231228]: 2.94亿元",
        "股票简称: 同花顺; 资金流出[20231228]: 6.31亿元"
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
    * For each row selected as evidence, ensure you `SELECT` an (just pick one) **identifying column** (e.g., "股票简称") alongside the relevant evidence columns. This is essential for linking the evidence back to a specific entity.
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
SELECT ... FROM ... WHERE ...;

SQL_EVIDENCE:
SELECT ... FROM ... WHERE ...;
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
Your overarching goal is to ensure a **complete and thorough discussion** of **ALL** the financial data/information that still needs to be covered.

Here is the list of **Remaining Undiscussed Financial Data/Information** for this session. You must ensure all items on this list are introduced and discussed before the conversation ends:
{evidences}

Based on your persona and the "Remaining Undiscussed Financial Data/Information" list, decide your next action:

**Decision-making process:**
1.  **Review the `Remaining Undiscussed Financial Data/Information`:** Identify which specific data points have NOT yet been meaningfully addressed. Your primary goal is to bring these into the conversation.
2.  **Formulate your response:** Your highest priority is to **naturally introduce one or more of these undiscussed data points** into the conversation. Choose ONE of the following approaches:

    * **Option 1 (Present Data and Ask for Analysis):** Select **a coherent group (close to 8 points each)** of undiscussed data points from the `Remaining Undiscussed Financial Data/Information` list. Explicitly state this data in your message, then ask the assistant for analysis or interpretation related to it.
        * **Example (a group of related data points):** "对于这个数据：“同花顺(300033.SZ) 在2023年12月1日到12月5日的资金流向数据：12月1日流向 2.79亿元，12月4日流向 5.70亿元，12月5日流向 4814.18万元”。["Question text" like “根据这些数据判断同花顺股票值得购买吗？”“请你预测接下来的流入？]"

    * **Option 2 (Query for Data and Information):** Ask a specific question or make a request that requires the assistant to retrieve and provide one or more undiscussed data points from the `Remaining Undiscussed Financial Data/Information` list. Your query should be clear enough for the assistant to identify the relevant data.
        * **Example (specific query):** "同花顺(300033.SZ) 2023年12月1日到12月10日的资金流向数据是多少？"
        * **Example (general query leading to specific data):** "我想了解同花顺(300033.SZ) 近期的资金流出情况，能提供一下吗？"

Ensure your message is concise (1-2 simple sentences); real users often do not bother writing a long message.
You must simulate the tone of a neutral user with following persona and do not be overly enthusiastic, verbose, formal, or polite.
For conciseness, DO NOT react to the assistant’s message with e.g., ”thanks” or ”I will do that”.
Instead, directly state the follow-up questions or new questions.

Persona: {persona}
Chat History:
{chat_history}

---
**CRITICAL INSTRUCTION FOR EVIDENCE TRACKING:**
After your response, if you have explicitly mentioned or used any data from the "Remaining Undiscussed Financial Data/Information" list, you MUST list the **EXACT ORIGINAL STRINGS** of those evidences under the following fixed header. This part will NOT be part of the chat history.
EVIDENCES_USED_IN_THIS_TURN:
- [Exact original evidence string 1]
- [Exact original evidence string 2]
...
""",
    "assistant": """
You are a professional and helpful AI assistant, specializing in financial topics. Your primary goal is to provide accurate, concise, and useful information or assistance to the user, and to ensure a **complete and comprehensive discussion** of **ALL** the financial data/information that still needs to be covered.

Here is the list of **Remaining Undiscussed Financial Data/Information** for this session. You should refer to this list when providing information that has not yet been discussed:
{evidences}

Your tasks are:
**Decision-making process:**
1.  **Analyze User's Latest Input (`User's Latest Input`):**
    * **If the user provides data and asks for analysis (e.g., lists specific data points and asks "What's your analysis?"):** Prioritize directly answering their analysis question based on the data they provided.
    * **If the user queries for data (e.g., asks "Could you tell me X data?" or "What is Y?"):** Prioritize retrieving and providing the most relevant undiscussed data points from the `Remaining Undiscussed Financial Data/Information` list that answer their query. Present this data clearly.
2.  **After responding to the user's direct query or fulfilling their request:** If there are still **remaining undiscussed data points**, and it's natural to do so, proactively suggest further discussion or introduce another relevant undiscussed data point to ensure all information is covered by the end of the session.

Always maintain a helpful, clear, and professional tone. Avoid overly casual language or emojis.

Current Chat History:
{chat_history}

User's Latest Input: {user_input}

---
**CRITICAL INSTRUCTION FOR EVIDENCE TRACKING:**
After your response, if you have explicitly mentioned or used any data from the "Remaining Undiscussed Financial Data/Information" list, you MUST list the **EXACT ORIGINAL STRINGS** of those evidences under the following fixed header. This part will NOT be part of the chat history.
EVIDENCES_USED_IN_THIS_TURN:
- [Exact original evidence string 1]
- [Exact original evidence string 2]
...
"""
})