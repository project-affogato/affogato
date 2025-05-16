import json

from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_json_string(json_string: str):
    # Remove code block markers and whitespace
    cleaned = (
        json_string.strip().removeprefix("```json").removesuffix("```").strip()
    )
    try:
        parsed_data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON string: {json_string}")
        parsed_data = "ERROR: JSON DECODE ERROR"

    return parsed_data
