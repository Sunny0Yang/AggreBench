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
    "evidence": ["Session <session_id>: Row <row_identifier>: <Column_Name>: <value>"] # Exactly 1 evidence entry. Use the most relevant row ID.
}}
5.  **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include the single, most direct evidence entry that is absolutely necessary to derive the answer.**
    - References MUST use "Session <session_id>: Row <row_identifier>: <Column_Name>: <value>" format.
    - The evidence provided must directly contain the value or data points used in the aggregation.
    - Must contain exactly 1 distinct evidence entry.
6.  **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For concise textual answers (e.g., MAX/MIN of text fields): "The answer is: [text]"
    - Avoid explanations or justifications in the answer field.

Example:
{{
    "question": "What is the latest stock price for 同花顺 (300033.SZ)?",
    "answer": "The answer is: 287.50",
    "evidence": ["Row 1: 最新价: 287.50元"]
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
    "evidence": ["Session <session_id>: Row <row_identifier>: <Column_Name>: <value>", ...] # 2-3 evidence entries.
}}
5.  **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include evidence that is absolutely necessary to derive the answer.**
    - If there aren't enough direct evidences to meet 'min_evidences' (2), *do not generate the question*.
    - References MUST use "Session <session_id>: Row <row_identifier>: <Column_Name>: <value>" format.
    - Quotes MUST be verbatim and directly support the answer's calculation or claim.
    - Each evidence entry MUST contribute directly to the calculation or understanding required to answer.
    - Must contain 2-3 distinct evidence entries.
6.  **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For concise textual answers: "The answer is: [text]"
    - Avoid explanations or justifications in the answer field.

Example:
{{
    "question": "What is the average daily capital inflow for 同花顺 (300033.SZ) in the first week of December 2023 (Dec 1st to Dec 7th)?",
    "answer": "The answer is: 263435420.0",
    "evidence": [
        "Row 1: 资金流向[20231201]: 2.79亿元",
        "Row 1: 资金流向[20231204]: 5.70亿元",
        "Row 1: 资金流向[20231205]: 4814.18万元",
        "Row 1: 资金流向[20231206]: 243.55万元",
        "Row 1: 资金流向[20231207]: 1.62亿元"
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
    "evidence": ["Session <session_id>: Row <row_identifier>: <Column_Name>: <value>", ...] # 4-5 evidence entries.
}}
6.  **Evidence Requirements**:
    - **CRITICAL: Evidence MUST be direct and highly relevant to the question's claim.**
    - **ONLY include evidence that is absolutely necessary to derive the answer.**
    - If there aren't enough direct evidences to meet 'min_evidences' (4), *do not generate the question*.
    - References MUST use "Session <session_id>: Row <row_identifier>: <Column_Name>: <value>" format.
    - Quotes MUST be verbatim and directly support the answer's calculation or claim.
    - Each evidence entry MUST contribute directly to the complex calculation or understanding required to answer.
    - Must contain 4-5 distinct evidence entries.
7.  **Answer Formatting**:
    - MUST begin with "The answer is: "
    - For numerical answers: "The answer is: [number]"
    - For concise textual answers: "The answer is: [text]"
    - Avoid explanations or justifications in the answer field.

Example:
{{
    "question": "Between December 1st and December 15th, 2023, what was the average *positive* daily capital inflow for 同花顺 (300033.SZ) after excluding days with negative flow?",
    "answer": "The answer is: 205696144.0",
    "evidence": [
        "Row 1: 资金流向[20231201]: 2.79亿元",
        "Row 1: 资金流向[20231204]: 5.70亿元",
        "Row 1: 资金流向[20231205]: 4814.18万元",
        "Row 1: 资金流向[20231206]: 243.55万元",
        "Row 1: 资金流向[20231207]: 1.62亿元",
        "Row 1: 资金流向[20231208]: 4.56亿元",
        "Row 1: 资金流向[20231211]: 3.57亿元",
        # Note: D12, D13, D14, D15 are negative or missing, not included as positive inflow evidence
    ]
}}
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
Your overarching goal is to ensure a **complete and thorough discussion** of **ALL** the financial data/information listed below. This data might include complex tables. Each "Table" or "Row" represents a distinct piece of information that needs to be addressed.
You must actively review the conversation history and compare it against the "All Required Evidences" list to determine what information has or has not yet been addressed.

Here is ALL the financial data/information that needs to be covered in this session:
{evidences}

Based on the current chat history, your persona, and the "All Required Evidences" list, decide your next action:

**Decision-making process:**
1.  **Analyze the `Chat History`:** Has any part of the `All Required Evidences` (including specific tables or rows like "Table 1, Row 0" or data points within them) been clearly discussed, mentioned, or implicitly covered?
2.  **Identify remaining undiscussed data:** Pinpoint which specific tables or rows from `All Required Evidences` have NOT yet been meaningfully addressed in the `Chat History`.
3.  **Formulate your response (prioritizing undiscussed data):**
    * If there are **undiscussed evidences (tables or rows)**: Your highest priority is to **naturally introduce one of these undiscussed pieces of data** into the conversation. Frame it as a question or statement. Choose one of the following approaches:
        * **Option 1 (Request specific row/data analysis):** Ask the assistant to provide or analyze a specific part of an undiscussed table/row.
            * **Example:** "Could you please list the **资金流向** for 同花顺 from 2023年12月1日 to 12月29日?" (This implies asking for all data in that specific row, or a summary of it.)
        * **Option 2 (Present full row data and ask question):** Provide a complete undiscussed row yourself and then ask a question about it.
            * **Example:** "I have the data for 同花顺's **资金流出** from Table 2, Row 0, which is: '股票代码: 300033.SZ, 股票简称: 同花顺, 最新价: 287.50元, 最新涨跌幅: -6.69%, 资金流出[20231201]: 5.04亿元, ...'. Given these figures, what's your analysis of the stock's future outlook?" (Note: The actual data string for the row would need to be inserted by your code when generating this part of the user's turn.)
    * If **all evidences seem discussed (or you believe they are sufficiently covered)**: Continue the conversation by asking follow-up questions, seeking clarification, or asking for a summary/analysis based on the discussed data.

Ensure your message is concise (1-2 simple sentences); real users often do not bother writing a long message.
You must simulate the tone of a neutral user and do not be overly enthusiastic, verbose, formal, or polite.
For conciseness, DO NOT react to the assistant’s message with e.g., ”thanks” or ”I will do that”.
Instead, directly state the follow-up questions or new questions.

Persona: {persona}
Chat History:
{chat_history}
""",
    "assistant": """
You are a professional and helpful AI assistant, specializing in financial topics. Your primary goal is to provide accurate, concise, and useful information or assistance to the user, and to ensure a **complete and comprehensive discussion** of **ALL** the financial data/information listed below. This data might include complex tables. Each "Table" or "Row" represents a distinct piece of information that needs to be addressed.
You must actively review the conversation history and compare it against the "All Required Evidences" list to determine what information has or has not yet been addressed.

Here is ALL the financial data/information that needs to be covered in this session:
{evidences}

Your tasks are:
**Decision-making process:**
1.  **First, directly answer the user's latest question or fulfill their request.** Prioritize responsiveness to the user's immediate input.
2.  **Next, analyze the `Chat History`:** Has any part of the `All Required Evidences` (including specific tables or rows like "Table 1, Row 0" or data points within them) been clearly discussed, mentioned, or implicitly covered?
3.  **Identify remaining undiscussed data:** Pinpoint which specific tables or rows from `All Required Evidences` have NOT yet been meaningfully addressed in the `Chat History`.
4.  **Formulate your response (prioritizing undiscussed data):**
    * If there are **undiscussed evidences (tables or rows)**: After responding to the user's direct query, **proactively introduce or prompt discussion about one of these undiscussed data points** if appropriate. Choose one of the following approaches:
        * **Option 1 (Provide specific row data directly):** Offer or provide a specific undiscussed row's data in response to a general query, or as a natural continuation.
            * **Example:** "Regarding 同花顺's performance, the detailed **资金流向** for December from Table 1, Row 0 is: '股票代码: 300033.SZ, 股票简称: 同花顺, 最新价: 287.50元, 最新涨跌幅: -6.69%, 资金流向[20231201]: 2.79亿元, ...'. This data indicates..." (Note: The actual data string for the row would need to be inserted by your code when generating this part of the assistant's turn.)
            * **Example:** "To give a complete overview, let's also examine 同花顺's **资金流出** figures from Table 2, Row 0, which are: '股票代码: 300033.SZ, 股票简称: 同花顺, 最新价: 287.50元, 最新涨跌幅: -6.69%, 资金流出[20231201]: 5.04亿元, ...'. How would you like to analyze these?"
        * **Option 2 (Suggest discussion of a table/row):** Ask the user if they'd like to delve into a specific undiscussed table or row.
            * **Example:** "We've discussed inflow. Would you like to review the **资金流出** for 同花顺 from Table 2 to get a comprehensive view?"
            * **Example:** "I also have detailed data on **同花顺's 资金流向**; would you like me to summarize the trends for specific periods, or analyze a particular date?"

Always maintain a helpful, clear, and professional tone. Avoid overly casual language or emojis.

Current Chat History:
{chat_history}

User's Latest Input: {user_input}

Please generate the next response from the assistant's perspective.
"""
})