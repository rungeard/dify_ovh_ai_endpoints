import importlib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

OpenAIText2SpeechModel = importlib.import_module("models.tts.tts").OpenAIText2SpeechModel


def test_strip_thinking_content_removes_leading_think_block() -> None:
    content = "<think>\nWe need to answer briefly.\n</think>Hello! 👋 How can I help you today?"

    assert OpenAIText2SpeechModel._strip_thinking_content(content) == (
        "Hello! 👋 How can I help you today?"
    )


def test_strip_thinking_content_keeps_plain_text_unchanged() -> None:
    content = "Hello! 👋 How can I help you today?"

    assert OpenAIText2SpeechModel._strip_thinking_content(content) == content
