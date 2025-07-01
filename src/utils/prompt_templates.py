from typing import Dict

QA_GENERATION_PROMPTS: Dict[str, str] =({
    "easy_aggregation_template_en": """
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

   "medium_aggregation_template_en": """
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

    "hard_aggregation_template_en": """
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
"""
})