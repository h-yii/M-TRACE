import re
import json
from typing import Any, Dict

class jsonParse:
    """
    Ultra-Robust parser for timeline structures produced by LLMs.
    It can handle:
    - Non-JSON brackets
    - Keys without quotes
    - Keys containing spaces
    - Missing commas
    - Single/double quotes mismatch
    - Extra text before/after structure
    - holder names without quotes
    """

    MAIN_PATTERN = re.compile(
        r'(\[.*\]|\{.*\})',
        flags=re.DOTALL
    )

    KEY_PATTERN = re.compile(
        r'([A-Za-z_][A-Za-z0-9_ ]*)\s*:',
        flags=re.DOTALL
    )

    BAREWORD_PATTERN = re.compile(
        r':\s*([A-Za-z][A-Za-z0-9 _\-]+)\s*([,\}])'
    )

    def extract_main_block(self, raw: str) -> str:
        m = self.MAIN_PATTERN.search(raw)
        if not m:
            raise ValueError("No structured block found in the text.")
        return m.group(1)

    def normalize_keys(self, text: str) -> str:
        def repl(match):
            key = match.group(1).strip()
            key = key.replace(" ", "_")
            return f'"{key}":'
        return self.KEY_PATTERN.sub(repl, text)

    def quote_barewords(self, text: str) -> str:
        def repl(match):
            value = match.group(1).strip()
            end = match.group(2)
            if value.startswith('"') and value.endswith('"'):
                return match.group(0)
            return f': "{value}"{end}'
        return self.BAREWORD_PATTERN.sub(repl, text)

    def ensure_json_braces(self, text: str) -> str:
        if text.strip().startswith("["):
            inner = text.strip()[1:-1]
            return "{" + inner + "}"
        return text
    
    def escape_inner_quotes(self, text: str) -> str:
        """
        Escape bad inner double quotes inside string values.
        e.g. "analysis": "foo "bar" baz" → "foo \"bar\" baz"
        """
        def fix_quotes(match):
            content = match.group(1)
            content = content.replace("\\", "\\\\")  
            content = re.sub(r'(?<!\\)"', r'\"', content)
            return f"\"{content}\""

        return re.sub(r'"([^"]*?)"', fix_quotes, text)

    def parse(self, raw: str) -> Dict[str, Any]:
        block = self.extract_main_block(raw)

        block = self.ensure_json_braces(block)
        block = self.normalize_keys(block)
        block = self.quote_barewords(block)
        block = self.escape_inner_quotes(block)

        block = re.sub(r',\s*}', "}", block)
        block = re.sub(r',\s*]', "]", block)

        try:
            print(block)
            return json.loads(block)
        except json.JSONDecodeError:
            print("DEBUG — final block looks like:")
            print(block)
            raise

