import pytest
from fastapi.testclient import TestClient

from main import get_application

EXPECTED_MODELS = {
    'tts-1',
    'tts-1-hd',
    'silero-tts-v5-ru',
    'silero-tts-v5-ukr',
}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv('TTS_ENGINE', 'mock')
    app = get_application()
    with TestClient(app) as test_client:
        yield test_client


def test_get_models_returns_expected_ids(client: TestClient):
    response = client.get('/v1/models')
    assert response.status_code == 200

    payload = response.json()
    assert payload['object'] == 'list'
    assert isinstance(payload['data'], list)

    model_ids = {item['id'] for item in payload['data']}
    assert EXPECTED_MODELS.issubset(model_ids)


def test_post_audio_speech_returns_streaming_audio(client: TestClient):
    request_payload = {
        'model': 'tts-1',
        'voice': 'xenia',
        'input': 'OpenAI compatible TTS smoke test',
    }

    with client.stream('POST', '/v1/audio/speech', json=request_payload) as response:
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('audio/')

        chunks = list(response.iter_bytes())
        assert chunks
        assert sum(len(chunk) for chunk in chunks) > 0


def test_post_audio_speech_unknown_voice_returns_400(client: TestClient):
    response = client.post(
        '/v1/audio/speech',
        json={
            'model': 'tts-1',
            'voice': 'unknown-voice',
            'input': 'test',
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload['error']['type'] == 'invalid_request_error'
    assert payload['error']['param'] == 'voice'


def test_post_audio_speech_unknown_response_format_returns_400(client: TestClient):
    response = client.post(
        '/v1/audio/speech',
        json={
            'model': 'tts-1',
            'voice': 'xenia',
            'input': 'test',
            'response_format': 'aac',
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload['error']['type'] == 'invalid_request_error'
    assert payload['error']['param'] == 'response_format'
