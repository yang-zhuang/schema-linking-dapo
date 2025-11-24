"""
Model interaction module
"""

import re
import json
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI
import json_repair
from typing import List, Dict

from .config import DEFAULT_REQUEST_TIMEOUT


class ModelClient:
    """Handles model interactions"""

    def __init__(self, api_url: str, api_key: str, model_name: str, system_prompt: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.client = None

    def init_client(self) -> OpenAI:
        """Initialize vLLM client"""
        try:
            self.client = OpenAI(base_url=self.api_url, api_key=self.api_key)
            return self.client
        except Exception as e:
            raise ConnectionError(f"LLM client initialization failed: {str(e)}")

    def call_model_single(self, prompt: str) -> str:
        """Call LLM model, get raw response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.1,  # Low temperature for stable output
                top_p=0.9,
                timeout=DEFAULT_REQUEST_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Model call failed: {str(e)}")
            return ""

    def call_model_batch(self, prompts: List[str], max_workers: int = 5) -> List[str]:
        """
        Batch call LLM model (concurrent processing)
        :param prompts: List of prompts
        :param max_workers: Number of concurrent workers
        :return: List of responses
        """
        def process_single(prompt):
            try:
                return self.call_model_single(prompt)
            except Exception as e:
                print(f"Single request failed: {str(e)}")
                return ""

        # Use thread pool for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_single, prompts),
                total=len(prompts),
                desc="Concurrent request progress"
            ))

        return results

    def parse_model_response(self, response_text: str) -> Dict:
        """Parse model response to JSON"""
        # Remove possible wrapper markers
        if "</think>" in response_text:
            match = re.search(r"</think>(.*?)$", response_text, re.DOTALL)
            if match:
                response_text = match.group(1).strip()

        # Repair JSON format and parse
        try:
            return json_repair.loads(response_text)
        except (json.JSONDecodeError, ValueError):
            print(f"Response parsing failed (first 100 chars): {response_text[:100]}...")
            return {"schema": []}  # Return empty structure to avoid interrupting flow

    def convert_pred_to_standard_str(self, pred_json: Dict) -> str:
        """Convert model prediction JSON to standardized string"""
        try:
            pred_schema = pred_json.get("schema", [])
        except:
            return "###Tables: ;\n###Columns: ;"

        if not isinstance(pred_schema, list):
            return "###Tables: ;\n###Columns: ;"

        pred_tables = []
        pred_cols = []
        for item in pred_schema:
            if not isinstance(item, dict):
                continue
            table = item.get("table_name")
            if not table:
                continue
            pred_tables.append(table)

            # Concatenate "table_name.column_name" format
            cols = item.get("columns", [])
            pred_cols.extend([f"{table}.{col}" for col in cols if isinstance(col, str)])

        tables_str = ', '.join(pred_tables)
        cols_str = ', '.join(pred_cols)
        return f"###Tables: {tables_str};\n###Columns: {cols_str};"