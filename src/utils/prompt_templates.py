from typing import Dict

QA_GENERATION_PROMPTS: Dict[str, str] =({
    "cross_session_template_en": """
Please generate an aggregative query question with practical business/workplace relevance based on these multiple sessions:

{session_context}

Generation Requirements:
1. **Core Operations**: MUST use counting, summing, or deduplication
2. **Scope**: MUST span at least {session_threshold} sessions
3. **Real-World Significance**: Questions should reveal:
   - Emerging problems/trends (e.g., recurring errors, sentiment shifts)
   - Resource needs (e.g., frequently requested features/tools)
   - Customer impact (e.g., affected user groups/regions)
   - Operational patterns (e.g., common request types, bottlenecks)
4. **Output Format**: Strict JSON only:
{{
    "question": "Question text",
    "answer": "Answer text",    
    "evidence": ["Evidence 1", "Evidence 2", ...], # Exactly {min_evidences}-{max_evidences} entries
}}
5. **Evidence Requirements**:
   - References MUST use "D{{session_id}}:{{turn_id}}" format
   - Quotes MUST be verbatim and relevant to the question's claim
   - Every evidence entry MUST contribute to demonstrating a root cause or solution context
   - Must contain at least {min_evidences} but no more than {max_evidences} distinct evidences

Example Output:
{{
    "question": "How many distinct customers reported payment failures in European transactions across these logs?",
    "answer": "5 customers",
    "evidence": [
        "D2:8: 'Payment failed for order EU-1092' - Customer ID 447",
        "D3:1: 'German client unable to process EUR payment' - Customer ID 589",
        "D4:3: 'French customer EU transaction declined' - Customer ID 312",
        "D5:6: 'Spain: Payment gateway error' - Customer ID 589",
        "D7:2: 'Italy: Credit card authorization failure' - Customer ID 732"
    ]
}}
"""
})