from typing import Dict

RESPONSE_GENERATION_PROMPTS: Dict[str, str] =({
    "template_en": """
Please generate an answer with concrete evidence based on the given memory context and the question.

Memory Context:
{memory_context}

Question:
{question}

Instructions:
1. Output Format MUST start with "The answer is: ".
2. Provide a concise final answer inside the sentence.
3. Do not use markdown or formatting.
4. If no sufficient information is available, respond with: "The answer is: unknown.", and briefly explain your reasoning.

Example Output:
The answer is: 42. Based on the calculation from the memory context, the result matches the known value of 42.
"""
})