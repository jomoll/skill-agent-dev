import hashlib
import json
import re
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from core_utils import is_reasoning_llm, setup_api_keys


class FHIRSampleEvaluator:
    def __init__(
        self,
        *,
        model: str,
        cache_path: Path,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, bool] = {}
        self._lock = threading.Lock()
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}
        setup_api_keys()

    def _cache_key(self, sample: Dict, answer: Optional[str], error: Optional[str]) -> str:
        payload = {
            "question_id": sample.get("question_id"),
            "question": sample.get("question"),
            "true_answer": sample.get("true_answer"),
            "answer": answer,
            "error": error,
            "model": self.model,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def score(self, sample: Dict, parsed_output: Dict) -> bool:
        answer = parsed_output.get("agent_answer")
        error = parsed_output.get("error")
        if error or answer is None:
            return False
        key = self._cache_key(sample, answer, error)
        with self._lock:
            if key in self.cache:
                return bool(self.cache[key])

        result = self._judge_answer(
            question=str(sample.get("question", "")),
            true_answer=str(sample.get("true_answer", "")),
            agent_answer=str(answer),
        )
        with self._lock:
            self.cache[key] = result
            self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        return result

    def _judge_answer(self, *, question: str, true_answer: str, agent_answer: str) -> bool:
        normalized = re.sub(r"[^\w\s\[\].:-]", "", agent_answer).strip().lower()
        if normalized == re.sub(r"[^\w\s\[\].:-]", "", true_answer).strip().lower():
            return True

        prompt = f"""You evaluate answers for FHIR patient-data questions.

Return only 1 if the model answer is semantically correct, otherwise only 0.

Be lenient about formatting, brackets, units, and explanatory text when the value is correct.
For yes/no answers, [[1]] means yes and [[0]] means no.
For null/no-answer cases, accept a clear statement that no matching data was found.
For numeric answers, ignore harmless decimal formatting.
For date/time answers, ignore timezone and formatting differences when the date/time meaning matches.

Question: {question}
True answer: {true_answer}
Model answer: {agent_answer}

Return 1 or 0."""
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=None if is_reasoning_llm(self.model) else 0.0,
                base_url=self.base_url,
                custom_llm_provider="openai" if self.base_url else None,
            )
            text = response.choices[0].message.content.strip()
            return text.startswith("1")
        except Exception as e:
            print(f"[FHIRSkillCycle] evaluator failed: {e}")
            return False
