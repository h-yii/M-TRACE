# action_decomposer.py
import os
from typing import List, Dict, Any
# from openai import OpenAI

import http.client
import json

class ActionDecomposer:

    def __init__(self, model="gpt-4o", temperature=0.2):
        self.model = model
        self.temperature = temperature


    def decompose(self, user_query: str):

        system_prompt1 = f"""
        You are a Precision Task Decomposer.
        Your goal is to break down a complex user query into a linear sequence of simple, atomic, and executable search actions.

        CORE PRINCIPLES:

        1. **Atomic Simplicity**:
        - Each step must represent exactly ONE search or reasoning action.
        - If a step involves multiple operations like "find X and then verify Y", split it into separate steps.

        2. **Explicit Temporal Specification**:
        - For temporal queries involving "before X" or "after Y", always explicitly retrieve specific dates.
        - Never use vague terms like "association", "involvement", or "connection" - specify "start date" or "end date".

        3. **Sequential Dependency**:
        - Design steps where Step N logically depends on the information obtained in Step N-1.
        - When query asks about "Entity A relative to Entity B", first fully resolve Entity B's timeline.

        4. **Comparative Precision**:
        - Temporal comparisons ("before", "after", "last before") require explicit date comparison steps.
        - "Last before X" means: find all entities ending before X's start, then select the one with latest end date.

        5. **Action-Oriented Language**:
        - Use precise verbs: "Retrieve", "Identify", "Compare", "Find", "List", "Extract".
        - Each step should be directly executable as a search query.

        ---

        ### TRANSFORMATION EXAMPLES

        **User Query:** "Who was the US President after the one who signed the Civil Rights Act of 1964?"

        **BAD Decomposition (vague):**
        - Find information about the president who signed the Civil Rights Act. (❌ Too vague)
        - Determine who came after him. (❌ Not atomic)

        **GOOD Decomposition:**
        [
            {{"step_id": 1, "description": "Retrieve the exact signing date of the Civil Rights Act of 1964."}},
            {{"step_id": 2, "description": "Identify the US President who was in office on the date found in Step 1."}},
            {{"step_id": 3, "description": "Retrieve the end date of that President's term."}},
            {{"step_id": 4, "description": "Find the US President whose term started immediately after the date found in Step 3."}}
        ]

        **User Query:** "Who was the last member of sports team of Barnsley F.C. before Aleš Križan? Options: A: Paul Wilkinson, B: Péter Rajczi, C: Tony Warner, D: Chris Pearce"

        **GOOD Decomposition:**
        [
            {{"step_id": 1, "description": "Retrieve Aleš Križan's start date as a member of Barnsley F.C."}},
            {{"step_id": 2, "description": "Retrieve Aleš Križan's end date as a member of Barnsley F.C."}},
            {{"step_id": 3, "description": "For candidate Paul Wilkinson, retrieve his start and end dates at Barnsley F.C."}},
            {{"step_id": 4, "description": "For candidate Péter Rajczi, retrieve his start and end dates at Barnsley F.C."}},
            {{"step_id": 5, "description": "For candidate Tony Warner, retrieve his start and end dates at Barnsley F.C."}},
            {{"step_id": 6, "description": "For candidate Chris Pearce, retrieve his start and end dates at Barnsley F.C."}},
            {{"step_id": 7, "description": "Identify which candidates have end dates that occur before Aleš Križan's start date."}},
            {{"step_id": 8, "description": "From the valid candidates in Step 7, determine which one has the latest end date."}}
        ]

        ---

        **Current User Query:**
        {user_query}

        Now, decompose this query into the most efficient, atomic executable steps:
        """
# =============================================

        conn = http.client.HTTPSConnection("...")
        payload = json.dumps({
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system", "content": system_prompt1
            }
        ]
        })
        headers = {
        'Accept': 'application/json',
        'Authorization': 'sk-xx',
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data_str = data.decode("utf-8") 
        data = json.loads(data_str)
        raw = data["choices"][0]["message"]["content"]
        print("Decomposer raw output:", raw)

# =============================================

        try:
            steps: List[Dict[str, Any]] = eval(raw)
        except:
            steps = [{"step_id": 1, "description": f"PARSE ERROR: {raw}"}]

        for i, step in enumerate(steps):
            step["step_id"] = i + 1
        print("Decomposed steps:", steps)
        return steps


if __name__ == "__main__":
    decomposer = ActionDecomposer()
    query = "Who was the head of government of Australia first before Robert Menzies?"
    decomposer.decompose(query)
