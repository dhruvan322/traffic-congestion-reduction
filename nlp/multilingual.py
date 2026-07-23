"""
Multilingual Module.
Language detection and translation for Hindi/Telugu/English.
"""

import logging

try:
    from langdetect import detect as _detect_lang
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """
    Detect the language of input text.
    Returns ISO 639-1 code: 'en', 'hi', 'te'.
    """
    if not HAS_LANGDETECT:
        return DEFAULT_LANGUAGE

    try:
        lang = _detect_lang(text)
        if lang in SUPPORTED_LANGUAGES:
            return lang
        return DEFAULT_LANGUAGE
    except Exception:
        return DEFAULT_LANGUAGE


def translate_text(text: str, source: str = "auto", target: str = "en") -> str:
    """
    Translate text between supported languages.
    Uses deep-translator (Google Translate backend).
    """
    if source == target:
        return text

    if not HAS_TRANSLATOR:
        logger.warning("deep-translator not installed. Returning original text.")
        return text

    try:
        translated = GoogleTranslator(source=source, target=target).translate(text)
        return translated or text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text


def get_language_name(code: str) -> str:
    """Get human-readable language name from ISO code."""
    names = {
        "en": "English",
        "hi": "Hindi (हिंदी)",
        "te": "Telugu (తెలుగు)",
    }
    return names.get(code, code.upper())
