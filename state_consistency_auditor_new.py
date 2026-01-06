# state_consistency_auditor.py
"""Structured World-State Consistency Auditor

This upgrades the auditor from returning free-form analysis strings to returning
structured conflict items that can be directly assembled into the new Conflict Report JSON
schema (see new_conflict_report.json).

Key additions:
- audit_structured(): returns (conflicts, extracted_facts) rather than (is_conflict, analysis)
- Backward-compatible audit(): kept for older code paths
"""

import json
import http.client
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonParse import jsonParse

POLOAI_HOST = "..."
POLOAI_PATH = "/v1/chat/completions"
POLOAI_MODEL = "gpt-4o"

API_KEY = 'sk-xx'

LOG_FILE = "test_auditor_log.txt"


def log(msg: Any) -> None:
    msg = str(msg)
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def call_gpt(messages, model: str = POLOAI_MODEL) -> str:
    """Internal LLM call used by the auditor."""
    if not API_KEY:
        raise RuntimeError("..")

    conn = http.client.HTTPSConnection(POLOAI_HOST)
    payload = json.dumps({
        "model": model,
        "messages": messages
    })

    headers = {
        "Accept": "application/json",
        "Authorization": API_KEY,
        "Content-Type": "application/json",
    }
    conn.request("POST", POLOAI_PATH, payload, headers)
    res = conn.getresponse()
    data = res.read()
    data_str = data.decode("utf-8")
    data = json.loads(data_str)
    raw = data["choices"][0]["message"]["content"]
    return raw


def _safe_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"null", "none", "n/a"}:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _normalize_time_span(t: Dict[str, Any]) -> Dict[str, Optional[int]]:
    return {"start": _safe_int(t.get("start")), "end": _safe_int(t.get("end"))}


def _dedup_facts(facts, external_knowledge) -> List[Dict[str, Any]]:
    external_knowledge = str(external_knowledge)
    seen = set()
    out = []
    for f in facts:
        s = f.get("s"); r = f.get("r"); o = f.get("o")
        if str(s) not in external_knowledge or o not in external_knowledge or r not in external_knowledge:
            continue
        t = f.get("t") or {}
        key = (s, r, o, _safe_int(t.get("start")), _safe_int(t.get("end")))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "fid": f.get("fid"),
            "s": s, "r": r, "o": o,
            "t": {"start": _safe_int(t.get("start")), "end": _safe_int(t.get("end"))}
        })
    return out


class StateConsistencyAuditor:
    """Checks each sub-action output against a world-state timeline (internal timeline).

    New API:
      audit_structured(step_output, source, step_meta=None) -> (conflicts, extracted_facts, parsed)
        - conflicts: List[conflict_item] matching the new conflict schema (subset)
        - extracted_facts: {"external_facts":[...], "internal_facts":[...]}
    """

    def __init__(self, world_state: Dict[str, Any], q_id: Any):
        self.world_state = world_state
        log("q_id in Auditor:" + str(q_id))
        self.json_parser = jsonParse()
        self._cid_counter = 0
        self._seen_conflict_keys = set()

    def _next_cid(self) -> str:
        self._cid_counter += 1
        return f"C{self._cid_counter}"

    def _world_state_graph_str(self) -> str:
        return json.dumps(self.world_state.get("timeline", []), ensure_ascii=False, indent=2)

    # ----------------------------
    # Backward compatible API
    # ----------------------------
    def audit(self, step_output: str, source: str):
        """Backward compatible auditor: returns (is_conflict, analysis, parsed)."""
        world_state_graph = self._world_state_graph_str()

        json_template = """{
            \"is_conflict\": true/false,
            \"analysis\": \"Explain the reasons in detail\"
        }"""

        system_prompt = ""

        messages = [{"role": "system", "content": system_prompt}]
        log("from:" + str(source))
        raw = call_gpt(messages)
        parsed = self.json_parser.parse(raw)
        try:
            return parsed["is_conflict"], parsed["analysis"], parsed
        except Exception:
            return None, "wrong", raw


    def checkC(self, c: dict) -> bool:
        """
        Check whether a conflict should be accepted.
        Deduplicate by (s, r, o).
        Return True if this conflict is new; False if duplicated.
        """
        key = c.get("key") or {}

        sig = (
            key.get("s"),
            key.get("r"),
            key.get("o"),
        )

        if sig in self._seen_conflict_keys:
            return False

        self._seen_conflict_keys.add(sig)
        return True



    # ----------------------------
    # New structured API
    # ----------------------------
    def audit_structured(
        self,
        step_output: str,
        source: str = "external",
        step_meta: Optional[Dict[str, Any]] = None,
        external_knowledge: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """Return structured conflicts + extracted facts.

        Returns
        -------
        conflicts:
            List of conflict objects (each with cid, type, key, external/internal spans, impact, severity).
        extracted:
            {"external_facts": [...], "internal_facts": [...]}
        parsed:
            Raw parsed JSON from auditor LLM.
        """
        step_meta = step_meta or {}
        world_state_graph = self._world_state_graph_str()
        world_state_name = self.world_state.get("world_state_name", self.world_state.get("world_state", "")) or ""

        json_template = """{
  \"consistent\": true/false,
  \"extracted_external_facts\": [
    {\"fid\":\"E1\",\"s\":\"...\",\"r\":\"...\",\"o\":\"...\",\"t\":{\"start\": 2000, \"end\": 2003}}
  ],
  \"extracted_internal_facts\": [
    {\"fid\":\"I1\",\"s\":\"<world_state>\",\"r\":\"<role_or_relation>\",\"o\":\"<holder>\",\"t\":{\"start\": 2000, \"end\": 2003}}
  ],
  \"conflicts\": [
    {
      \"type\": \"time_span_mismatch | holder_mismatch | sequence_violation | overlap_violation | other\",
      \"key\": {\"s\":\"...\",\"r\":\"...\",\"o\":\"...\"},
      \"external\": {\"ref_fids\":[\"E1\"], \"t\":{\"start\": 2000, \"end\": 2003}},
      \"internal\": {\"ref_fids\":[\"I1\"], \"t\":{\"start\": 2000, \"end\": 2003}},
      \"impact\": {\"affects_time_anchor\": true/false, \"affects_option_ranking\": true/false},
      \"severity\": 0.0
    }
  ]
}"""

        system_prompt = f"""You are a Structured World-State Consistency Auditor.

You are given:
1) step_output (claims produced by an agent, usually based on EXTERNAL knowledge)
2) world_state_graph (INTERNAL world-state timeline for a SINGLE world state)

Your tasks:
A. Extract atomic EXTERNAL facts from step_output when they mention a time span, holder, or relation.
B. Convert world_state_graph timeline entries into atomic INTERNAL facts (one fact per timeline segment).
C. Detect explicit logical/temporal conflicts between extracted_external_facts and extracted_internal_facts.

Conflict criteria:
- Only mark conflict when the same key (s,r,o) cannot both be true at the same time.
- Do NOT treat missing facts in world_state_graph as contradiction.
- Prefer explicit temporal incompatibility (overlap with different holders, mismatched spans for same triple, impossible sequence).

World-state name:
{world_state_name}

world_state_graph (timeline JSON):
{world_state_graph}

step_output:
{step_output}

Output requirement:
- Output MUST be valid JSON and MUST strictly follow the template below.
- If no conflicts, set conflicts=[] and consistent=true.
- severity is in [0,1]. Use higher values for anchor-affecting or option-affecting conflicts.

JSON TEMPLATE:
{json_template}
""".strip()

        messages = [{"role": "system", "content": system_prompt}]
        log(f"from:{source}")
        raw = call_gpt(messages)
        parsed = self.json_parser.parse(raw)

        conflicts: List[Dict[str, Any]] = []
        for c in (parsed.get("conflicts") or []):
            cid = self._next_cid()
            key = c.get("key") or {}
            external = c.get("external") or {}
            internal = c.get("internal") or {}
            external_knowledge = str(external_knowledge)
            if key.get("s") is None or key.get("r") is None or key.get("o") is None:
                continue
            if key.get("s") not in external_knowledge or key.get("o") not in external_knowledge or key.get("r") not in external_knowledge:
                continue
            if not self.checkC(c):
                continue


            conflicts.append({
                "cid": cid,
                "type": c.get("type", "other"),
                "key": {"s": key.get("s"), "r": key.get("r"), "o": key.get("o")},
                "external": {
                    "ref_fids": external.get("ref_fids", []),
                    "t": _normalize_time_span(external.get("t") or {})
                },
                "internal": {
                    "ref_fids": internal.get("ref_fids", []),
                    "t": _normalize_time_span(internal.get("t") or {})
                },
                "impact": c.get("impact") or {"affects_time_anchor": False, "affects_option_ranking": False},
                "severity": float(c.get("severity", 0.0) or 0.0),
                "meta": {"source": source, **step_meta}
            })

        extracted = {
            "external_facts": _dedup_facts(parsed.get("extracted_external_facts" or []), external_knowledge),
            "internal_facts": _dedup_facts(parsed.get("extracted_internal_facts" or []), external_knowledge),
        }
        return conflicts, extracted, parsed

    def export_report(self, path: Path):
        """Legacy export (kept for compatibility)."""
        report = {
            "question": self.world_state.get("question", ""),
            "timeline": self.world_state.get("timeline", []),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
