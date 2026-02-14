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


def test_post_audio_speech_put_accent_must_be_boolean(client: TestClient):
    response = client.post(
        '/v1/audio/speech',
        json={
            'model': 'tts-1',
            'voice': 'xenia',
            'input': 'test',
            'put_accent': 'yes',
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload['error']['type'] == 'invalid_request_error'
    assert payload['error']['param'] == 'put_accent'


def test_post_audio_speech_forwards_homograph_and_accent_flags(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    captured_kwargs = {}

    class SpyEngine:
        def synthesize(
            self,
            model_name: str,
            voice: str,
            text: str,
            response_format: str,
            sample_rate: int,
            speed: float,
            bitrate: object | None = None,
            put_accent: bool = True,
            put_yo: bool = True,
            put_stress_homo: bool = True,
            put_yo_homo: bool = True,
        ) -> bytes:
            captured_kwargs.update(
                {
                    'put_accent': put_accent,
                    'put_yo': put_yo,
                    'put_stress_homo': put_stress_homo,
                    'put_yo_homo': put_yo_homo,
                }
            )
            return b'FAKE_AUDIO'

    monkeypatch.setattr('app.handlers.get_tts_engine_dependency', lambda: SpyEngine())

    response = client.post(
        '/v1/audio/speech',
        json={
            'model': 'tts-1',
            'voice': 'xenia',
            'input': 'Пот+ом замок откр+оется.',
            'response_format': 'wav',
            'put_accent': False,
            'put_yo': False,
            'put_stress_homo': False,
            'put_yo_homo': False,
        },
    )

    assert response.status_code == 200
    assert captured_kwargs == {
        'put_accent': False,
        'put_yo': False,
        'put_stress_homo': False,
        'put_yo_homo': False,
    }
