import logging
import os
from collections.abc import Iterator
from functools import lru_cache
from typing import Any, Protocol

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.tts_engine import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEED,
    MAX_SPEED,
    MIN_SPEED,
    OPENAI_MODEL_IDS,
    OPENAI_RESPONSE_FORMATS,
    MockTTSEngine,
    SileroEngineError,
    SileroTTSEngine,
)

logger = logging.getLogger('uvicorn')
router = APIRouter()
v1_router = APIRouter(prefix='/v1')


class TTSEngineProtocol(Protocol):
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
    ) -> bytes: ...


@lru_cache
def get_tts_engine() -> SileroTTSEngine:
    return SileroTTSEngine()


@lru_cache
def get_mock_tts_engine() -> MockTTSEngine:
    return MockTTSEngine()


def get_tts_engine_dependency() -> TTSEngineProtocol:
    engine_name = os.getenv('TTS_ENGINE', '').strip().lower()
    if engine_name == 'mock':
        return get_mock_tts_engine()
    return get_tts_engine()


def _openai_error_response(message: str, param: str | None, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            'error': {
                'message': message,
                'type': 'invalid_request_error',
                'param': param,
                'code': None,
            }
        },
    )


def _chunk_bytes(data: bytes, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    for index in range(0, len(data), chunk_size):
        yield data[index : index + chunk_size]


def _parse_non_empty_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SileroEngineError(f"Parameter '{key}' must be a non-empty string.", key, 422)
    return value


def _parse_response_format(payload: dict[str, Any]) -> str:
    response_format = payload.get('response_format', 'wav')
    if not isinstance(response_format, str):
        raise SileroEngineError(
            "Parameter 'response_format' must be a string.", 'response_format', 422
        )
    response_format = response_format.lower()
    if response_format not in OPENAI_RESPONSE_FORMATS:
        raise SileroEngineError(
            'Unsupported response_format. Use one of: wav, mp3, opus, flac, pcm.',
            'response_format',
            400,
        )
    return response_format


def _parse_speed(payload: dict[str, Any]) -> float:
    speed = payload.get('speed', DEFAULT_SPEED)
    try:
        speed_value = float(speed)
    except (TypeError, ValueError) as exc:
        raise SileroEngineError("Parameter 'speed' must be a number.", 'speed', 422) from exc
    if speed_value < MIN_SPEED or speed_value > MAX_SPEED:
        raise SileroEngineError("Parameter 'speed' must be between 0.25 and 4.0.", 'speed', 400)
    return speed_value


def _parse_sample_rate(payload: dict[str, Any]) -> int:
    sample_rate = payload.get('sample_rate', DEFAULT_SAMPLE_RATE)
    try:
        sample_rate_value = int(sample_rate)
    except (TypeError, ValueError) as exc:
        raise SileroEngineError(
            "Parameter 'sample_rate' must be an integer.",
            'sample_rate',
            422,
        ) from exc
    if sample_rate_value <= 0:
        raise SileroEngineError(
            "Parameter 'sample_rate' must be greater than 0.",
            'sample_rate',
            400,
        )
    return sample_rate_value


def _parse_optional_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise SileroEngineError(f"Parameter '{key}' must be a boolean.", key, 422)


def _parse_speech_payload(
    payload: Any,
) -> tuple[str, str, str, str, int, float, object | None, bool, bool, bool, bool]:
    if not isinstance(payload, dict):
        raise SileroEngineError('JSON body must be an object.', None, 422)

    for required_param in ('model', 'voice', 'input'):
        if required_param not in payload:
            raise SileroEngineError(
                f"Missing required parameter: '{required_param}'.",
                required_param,
                422,
            )

    model = _parse_non_empty_string(payload, 'model')
    if model not in OPENAI_MODEL_IDS:
        raise SileroEngineError(f"Model '{model}' not found.", 'model', 400)

    voice = _parse_non_empty_string(payload, 'voice')
    input_text = _parse_non_empty_string(payload, 'input')
    response_format = _parse_response_format(payload)
    sample_rate = _parse_sample_rate(payload)
    speed = _parse_speed(payload)
    bitrate = payload.get('bitrate')
    put_accent = _parse_optional_bool(payload, 'put_accent', True)
    put_yo = _parse_optional_bool(payload, 'put_yo', True)
    put_stress_homo = _parse_optional_bool(payload, 'put_stress_homo', True)
    put_yo_homo = _parse_optional_bool(payload, 'put_yo_homo', True)
    return (
        model,
        voice,
        input_text,
        response_format,
        sample_rate,
        speed,
        bitrate,
        put_accent,
        put_yo,
        put_stress_homo,
        put_yo_homo,
    )


@router.get('/health')
async def health():
    return {'status': 'ok'}


@v1_router.get('/models')
async def list_models():
    models = [
        {
            'id': model_id,
            'object': 'model',
            'created': 0,
            'owned_by': 'silero',
        }
        for model_id in OPENAI_MODEL_IDS
    ]
    return {'object': 'list', 'data': models}


@v1_router.post('/audio/speech')
async def create_audio_speech(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return _openai_error_response('Invalid JSON body.', None, 400)

    try:
        (
            model,
            voice,
            input_text,
            response_format,
            sample_rate,
            speed,
            bitrate,
            put_accent,
            put_yo,
            put_stress_homo,
            put_yo_homo,
        ) = _parse_speech_payload(payload)
        tts_engine = get_tts_engine_dependency()
    except SileroEngineError as error:
        return _openai_error_response(error.message, error.param, error.status_code)

    try:
        audio_bytes = tts_engine.synthesize(
            model_name=model,
            voice=voice,
            text=input_text,
            response_format=response_format,
            sample_rate=sample_rate,
            speed=speed,
            bitrate=bitrate,
            put_accent=put_accent,
            put_yo=put_yo,
            put_stress_homo=put_stress_homo,
            put_yo_homo=put_yo_homo,
        )
    except SileroEngineError as error:
        return _openai_error_response(error.message, error.param, error.status_code)
    except Exception as exception:
        logger.exception('Failed to generate audio: %s', exception)
        return _openai_error_response('Failed to generate audio.', None, 400)

    return StreamingResponse(
        _chunk_bytes(audio_bytes),
        media_type=OPENAI_RESPONSE_FORMATS[response_format],
    )


router.include_router(v1_router)
