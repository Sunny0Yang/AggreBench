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