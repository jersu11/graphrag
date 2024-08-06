# Licensed under the MIT License

import json
import re

"""JSON cleaning and formatting utilities."""

def clean_up_json(json_str: str):
    """Clean up json string."""
    json_str = (
        json_str.replace("\\n", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if json_str.startswith("```json"):
        json_str = json_str[len("```json") :]
    if json_str.startswith("json"):
        json_str = json_str[len("json") :]
    if json_str.endswith("```"):
        json_str = json_str[: len(json_str) - len("```")]

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        json_obj = extract_json_object(json_str)
        if json_obj is not None:
            return json.dumps(json_obj)

    return json_str


def fix_json_structure(json_str):
    # Fix missing commas between array elements
    json_str = re.sub(r'}\s*{', '},{', json_str)
    
    # Fix unescaped quotes within string values
    json_str = re.sub(r'(?<=: ")(.+?)(?="[,}])', lambda m: m.group(1).replace('"', '\\"'), json_str)
    
    return json_str

def extract_json_object(raw_str):
    # Use a more compatible regex to extract JSON
    json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    json_match = re.search(json_pattern, raw_str, re.DOTALL)
    if not json_match:
        #log.warning(f"No valid JSON object found in the input string --- {raw_str}")
        return None

    json_str = json_match.group(0)

    try:
        # Attempt to parse the JSON
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError as e:
        # last try. attempt a fix on the structure
        try:
            fixed_json_str = fix_json_structure(json_str)
            json_obj = json.loads(fixed_json_str)
            return json_obj
        except json.JSONDecodeError as e:
            #log.info(f"After RegEx load, still a JSON parsing error: {str(e)}")
            return None