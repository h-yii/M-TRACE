# reason.py
import http.client
import json


class BaseReasoner:

    def __init__(self, model_name="gpt-4o", api_key="", host="..."):
        self.model_name = model_name
        self.api_key = 'sk-xx'
        self.host = host

    def _call_gpt(self, system_prompt: str, user_prompt: str = "") -> str:

        return ""


# ============================================================
#                     Internal Reasoner
# ============================================================
class InternalReasoner(BaseReasoner):

    def run(self, sub_action: str) -> str:

        system_prompt = f"""
        You are the INTERNAL EXECUTION AGENT.
        You must complete the reasoning step ONLY using internal parametric knowledge.
        You cannot reference external files, retrieved knowledge, URLs, memories, or anything the user provides.

        Your task:
        - Execute the reasoning step faithfully.
        - Produce a concise, factual result.
        - If the information is uncertain, state uncertainty explicitly.

        Sub-action to execute:
        {sub_action}
        """

        result = self._call_gpt(system_prompt)
        return result


# ============================================================
#                     External Reasoner
# ============================================================
class ExternalReasoner(BaseReasoner):

    def run0(self, sub_action: str, external_knowledge: str) -> str:

        system_prompt = f"""
        You are the EXTERNAL EXECUTION AGENT.
        You must complete the reasoning step ONLY using the external knowledge provided below.
        You must ignore internal knowledge, ignore your own memory, and rely solely on the given text.

        External Knowledge:
        \"\"\"
        {external_knowledge}
        \"\"\"

        Your task:
        - Use ONLY the text above to execute the sub-action.
        - Do NOT add facts not present in the external knowledge.

        Sub-action to execute:
        {sub_action}
        """

        result = self._call_gpt(system_prompt)
        return result

    def run(self, sub_action: str, external_knowledge: str, original_question: str = "") -> tuple[str, str]:
        system_prompt = f"""
        You are the EXTERNAL EXECUTION AGENT.
        Your task is to execute a specific sub-action to help answer the original question.
        
        # Original Question:
        {original_question}
        
        # Sub-action to Execute:
        {sub_action}
        
        # External Knowledge (MUST use ONLY this knowledge):
        \"\"\"
        {external_knowledge}
        \"\"\"
        
        # Instructions:
        1. FIRST, understand the original question context
        2. THEN, execute ONLY the specified sub-action using ONLY the external knowledge
        3. Your reasoning MUST be grounded in the external knowledge provided
        4. Do NOT use your own internal knowledge or memories
        5. Do NOT add facts not present in the external knowledge
        6. If the external knowledge does not contain enough information, state what is missing
        
        # Output Format:
        Reasoning Process:
        [Your step-by-step reasoning here, showing how you use the external knowledge to execute the sub-action]
        
        Result:
        [Your concise answer to the sub-action based ONLY on external knowledge]
        """
        
        raw_output = self._call_gpt(system_prompt)
        reasoning_process = ""
        result = ""
        if "Reasoning Process:" in raw_output and "Result:" in raw_output:
            reasoning_part, result_part = raw_output.split("Result:", 1)
            reasoning_process = reasoning_part.split("Reasoning Process:", 1)[1].strip()
            result = result_part.strip()
        else:
            result = raw_output.strip()
            reasoning_process = "No explicit reasoning process recorded."
        return result, reasoning_process