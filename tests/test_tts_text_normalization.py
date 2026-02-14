from app.tts_engine import normalize_tts_text


def test_normalize_tts_text_converts_newlines_to_pauses():
    text = 'Привет\nкак дела\n\nВсе хорошо'
    assert normalize_tts_text(text) == 'Привет, как дела. Все хорошо.'


def test_normalize_tts_text_handles_markdown_blocks_with_strong_pauses():
    text = '# План\n- Первый пункт\n- Второй пункт\nИтог'
    assert normalize_tts_text(text) == 'План. Первый пункт. Второй пункт. Итог.'
