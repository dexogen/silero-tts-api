import io
import logging
import math
import os
import random
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from silero import silero_tts
except ImportError:
    silero_tts = None

logger = logging.getLogger('uvicorn')


OPENAI_MODEL_IDS = (
    'tts-1',
    'tts-1-hd',
    'silero-tts-v5-ru',
    'silero-tts-v5-ukr',
)

OPENAI_RESPONSE_FORMATS = {
    'wav': 'audio/wav',
    'mp3': 'audio/mpeg',
    'opus': 'audio/ogg',
    'flac': 'audio/flac',
    'pcm': 'application/octet-stream',
}

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_SPEED = 1.0
MIN_SPEED = 0.25
MAX_SPEED = 4.0
SUPPORTED_SAMPLE_RATES = {8000, 24000, 48000}
SUPPORTED_BITRATES = {64, 96, 128, 192}

RU_VOICES = ('aidar', 'baya', 'kseniya', 'xenia', 'eugene')
UKR_VOICES = (
    'ukr_mykyta',
    'ukr_kateryna',
    'ukr_lada',
    'ukr_oleksa',
    'ukr_tetiana',
)

VOICE_ALIASES = {
    'ru': {
        'aidar': 'aidar',
        'baya': 'baya',
        'kseniya': 'kseniya',
        'xenia': 'xenia',
        'eugene': 'eugene',
        'random': 'random',
    },
    'ukr': {
        'mykyta': 'ukr_mykyta',
        'ukr_mykyta': 'ukr_mykyta',
        'kateryna': 'ukr_kateryna',
        'ukr_kateryna': 'ukr_kateryna',
        'lada': 'ukr_lada',
        'ukr_lada': 'ukr_lada',
        'oleksa': 'ukr_oleksa',
        'ukr_oleksa': 'ukr_oleksa',
        'tetiana': 'ukr_tetiana',
        'ukr_tetiana': 'ukr_tetiana',
        'random': 'random',
    },
}

STRONG_PAUSE_PUNCTUATION = ('.', '!', '?', '…')
ANY_PAUSE_PUNCTUATION = STRONG_PAUSE_PUNCTUATION + (',', ';', ':')


def _is_markdown_block_start(line: str) -> bool:
    return bool(
        re.match(r'^\s{0,3}#{1,6}\s+', line) or re.match(r'^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)', line)
    )


def _strip_markdown_prefix(line: str) -> str:
    line = re.sub(r'^\s{0,3}#{1,6}\s+', '', line)
    line = re.sub(r'^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)', '', line)
    return line.strip()


def normalize_tts_text(text: str) -> str:
    normalized = text.replace('\r\n', '\n').replace('\r', '\n').replace('\t', ' ')
    lines = normalized.split('\n')

    fragments: list[str] = []
    paragraph_break = False
    previous_was_markdown_block = False
    for raw_line in lines:
        line_is_markdown_block = _is_markdown_block_start(raw_line)
        line = _strip_markdown_prefix(raw_line)
        if not line:
            paragraph_break = True
            continue

        if fragments:
            previous = fragments[-1].rstrip()
            if paragraph_break or line_is_markdown_block or previous_was_markdown_block:
                if not previous.endswith(STRONG_PAUSE_PUNCTUATION):
                    fragments[-1] = f'{previous}.'
            elif not previous.endswith(ANY_PAUSE_PUNCTUATION):
                fragments[-1] = f'{previous},'

        fragments.append(line)
        paragraph_break = False
        previous_was_markdown_block = line_is_markdown_block

    if not fragments:
        return text.strip()

    merged = re.sub(r'\s+', ' ', ' '.join(fragments)).strip()
    if merged and not merged.endswith(STRONG_PAUSE_PUNCTUATION):
        merged = f'{merged}.'
    return merged


@dataclass(frozen=True)
class ModelMapping:
    language: str
    model_id: str
    load_language: str


MODEL_MAPPINGS: dict[str, ModelMapping] = {
    'tts-1': ModelMapping(language='ru', model_id='v5_ru', load_language='ru'),
    'tts-1-hd': ModelMapping(language='ru', model_id='v5_ru', load_language='ru'),
    'silero-tts-v5-ru': ModelMapping(language='ru', model_id='v5_ru', load_language='ru'),
    'silero-tts-v5-ukr': ModelMapping(language='ukr', model_id='v5_cis_ext', load_language='ru'),
}


class SileroEngineError(Exception):
    def __init__(self, message: str, param: str | None = None, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.param = param
        self.status_code = status_code


class MockTTSEngine:
    @staticmethod
    def _resolve_model_mapping(model: str) -> ModelMapping:
        mapping = MODEL_MAPPINGS.get(model)
        if mapping is None:
            raise SileroEngineError(f"Model '{model}' not found.", 'model', 400)
        return mapping

    @staticmethod
    def _resolve_voice(language: str, voice: str) -> str:
        aliases = VOICE_ALIASES.get(language, {})
        normalized = voice.strip().lower()
        resolved = aliases.get(normalized)
        if resolved is None:
            supported = sorted(aliases.keys())
            raise SileroEngineError(
                f"Voice '{voice}' is not supported for language '{language}'. Supported voices: {', '.join(supported)}.",
                'voice',
                400,
            )

        if resolved == 'random':
            if language == 'ru':
                return RU_VOICES[0]
            return UKR_VOICES[0]

        return resolved

    @staticmethod
    def _normalize_bitrate(response_format: str, bitrate: object | None) -> str | None:
        if response_format not in {'mp3', 'opus'}:
            return None

        if bitrate is None:
            return None

        if isinstance(bitrate, str):
            normalized = bitrate.strip().lower()
            if not normalized:
                return None
            normalized = normalized.removesuffix('kbps').removesuffix('k').strip()
            if not normalized.isdigit():
                raise SileroEngineError(
                    "Parameter 'bitrate' must be one of: 64, 96, 128, 192.",
                    'bitrate',
                    400,
                )
            bitrate_value = int(normalized)
        elif isinstance(bitrate, int | float):
            if isinstance(bitrate, float) and not bitrate.is_integer():
                raise SileroEngineError(
                    "Parameter 'bitrate' must be one of: 64, 96, 128, 192.",
                    'bitrate',
                    400,
                )
            bitrate_value = int(bitrate)
        else:
            raise SileroEngineError(
                "Parameter 'bitrate' must be one of: 64, 96, 128, 192.",
                'bitrate',
                400,
            )

        if bitrate_value not in SUPPORTED_BITRATES:
            raise SileroEngineError(
                "Parameter 'bitrate' must be one of: 64, 96, 128, 192.",
                'bitrate',
                400,
            )

        return str(bitrate_value)

    @staticmethod
    def _mock_wav(
        sample_rate: int, frequency_hz: float = 440.0, duration_seconds: float = 0.06
    ) -> bytes:
        total_samples = max(1, int(sample_rate * duration_seconds))
        frames = bytearray()
        for sample_index in range(total_samples):
            sample_value = int(
                32767 * 0.15 * math.sin(2 * math.pi * frequency_hz * sample_index / sample_rate)
            )
            frames.extend(struct.pack('<h', sample_value))

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(frames))
        return buffer.getvalue()

    @staticmethod
    def _mock_pcm(sample_rate: int, speed: float) -> bytes:
        wav_audio = MockTTSEngine._mock_wav(sample_rate=sample_rate)
        with wave.open(io.BytesIO(wav_audio), 'rb') as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
        if speed == 1.0:
            return pcm

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size <= 1:
            return pcm
        new_size = max(1, int(samples.shape[0] / speed))
        source_index = np.arange(samples.shape[0], dtype=np.float32)
        target_index = np.linspace(0, samples.shape[0] - 1, num=new_size, dtype=np.float32)
        adjusted = np.interp(target_index, source_index, samples).astype(np.int16)
        return adjusted.tobytes()

    def synthesize(
        self,
        model_name: str,
        voice: str,
        text: str,
        response_format: str,
        sample_rate: int,
        speed: float,
        bitrate: object | None = None,
    ) -> bytes:
        normalized_text = normalize_tts_text(text)
        if not normalized_text:
            raise SileroEngineError("Parameter 'input' must be a non-empty string.", 'input', 422)

        mapping = self._resolve_model_mapping(model_name)
        self._resolve_voice(mapping.language, voice)

        if response_format not in OPENAI_RESPONSE_FORMATS:
            raise SileroEngineError(
                'Unsupported response_format. Use one of: wav, mp3, opus, flac, pcm.',
                'response_format',
                400,
            )

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise SileroEngineError(
                'Unsupported sample_rate. Use one of: 8000, 24000, 48000.',
                'sample_rate',
                400,
            )

        if speed < MIN_SPEED or speed > MAX_SPEED:
            raise SileroEngineError("Parameter 'speed' must be between 0.25 and 4.0.", 'speed', 400)

        self._normalize_bitrate(response_format=response_format, bitrate=bitrate)

        if response_format == 'pcm':
            return self._mock_pcm(sample_rate=sample_rate, speed=speed)

        wav_audio = self._mock_wav(sample_rate=sample_rate)
        if response_format == 'wav':
            return wav_audio

        # Mock output for encoded formats in tests without external tools.
        return f'MOCK-{response_format}-AUDIO'.encode('ascii') + wav_audio[:128]


class SileroTTSEngine:
    def __init__(self):
        if torch is None:
            raise SileroEngineError(
                'PyTorch is required for the Silero engine but is not installed.',
                None,
                500,
            )

        self._models: dict[tuple[str, str], object] = {}
        self._models_lock = threading.Lock()
        self._device = self._resolve_device()
        self._configure_num_threads()

    @staticmethod
    def _resolve_device() -> object:
        requested = os.getenv('SILERO_DEVICE', 'auto').strip().lower()

        if requested not in {'auto', 'cpu', 'cuda'}:
            raise SileroEngineError(
                'Invalid SILERO_DEVICE value. Use one of: auto, cpu, cuda.',
                'SILERO_DEVICE',
                400,
            )

        if requested == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if requested == 'cuda' and not torch.cuda.is_available():
            raise SileroEngineError(
                "SILERO_DEVICE is set to 'cuda', but CUDA is not available.",
                'SILERO_DEVICE',
                400,
            )

        return torch.device(requested)

    @staticmethod
    def _configure_num_threads() -> None:
        raw_value = os.getenv('NUMBER_OF_THREADS', '4').strip()
        try:
            num_threads = int(raw_value)
        except ValueError as exc:
            raise SileroEngineError(
                'NUMBER_OF_THREADS must be a positive integer.',
                'NUMBER_OF_THREADS',
                400,
            ) from exc

        if num_threads <= 0:
            raise SileroEngineError(
                'NUMBER_OF_THREADS must be greater than zero.',
                'NUMBER_OF_THREADS',
                400,
            )

        torch.set_num_threads(num_threads)

    def _resolve_model_mapping(self, model: str) -> ModelMapping:
        mapping = MODEL_MAPPINGS.get(model)
        if mapping is None:
            raise SileroEngineError(f"Model '{model}' not found.", 'model', 400)
        return mapping

    @staticmethod
    def _get_random_voice(language: str) -> str:
        if language == 'ru':
            return random.choice(RU_VOICES)
        if language == 'ukr':
            return random.choice(UKR_VOICES)
        raise SileroEngineError(f"Language '{language}' is not supported.", 'voice', 400)

    def resolve_voice(self, language: str, voice: str) -> str:
        aliases = VOICE_ALIASES.get(language, {})
        normalized = voice.strip().lower()
        resolved = aliases.get(normalized)

        if resolved is None:
            supported = sorted(aliases.keys())
            raise SileroEngineError(
                f"Voice '{voice}' is not supported for language '{language}'. Supported voices: {', '.join(supported)}.",
                'voice',
                400,
            )

        if resolved == 'random':
            return self._get_random_voice(language)

        return resolved

    def _load_model(self, mapping: ModelMapping):
        key = (mapping.load_language, mapping.model_id)
        with self._models_lock:
            cached = self._models.get(key)
            if cached is not None:
                return cached

            if silero_tts is not None:
                model, _ = silero_tts(language=mapping.load_language, speaker=mapping.model_id)
            else:
                model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=mapping.load_language,
                    speaker=mapping.model_id,
                )

            if hasattr(model, 'to'):
                model.to(self._device)
            if hasattr(model, 'eval'):
                model.eval()
            self._models[key] = model
            return model

    @staticmethod
    def _serialize_wav(audio_pcm: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_pcm)
        return buffer.getvalue()

    @staticmethod
    def _resample_for_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        if speed == 1.0 or audio.size <= 1:
            return audio

        new_size = max(1, int(audio.shape[0] / speed))
        source_index = np.arange(audio.shape[0], dtype=np.float32)
        target_index = np.linspace(0, audio.shape[0] - 1, num=new_size, dtype=np.float32)
        return np.interp(target_index, source_index, audio).astype(np.float32)

    @staticmethod
    def _find_sox_binary() -> str | None:
        return shutil.which('sox')

    @staticmethod
    def _find_opusenc_binary() -> str | None:
        return shutil.which('opusenc')

    @classmethod
    def _require_sox_binary(cls, response_format: str) -> str:
        sox_binary = cls._find_sox_binary()
        if sox_binary is None:
            raise SileroEngineError(
                f"response_format '{response_format}' requires 'sox' to be installed and available in PATH.",
                'response_format',
                400,
            )
        return sox_binary

    @classmethod
    def _require_opusenc_binary(cls) -> str:
        opusenc_binary = cls._find_opusenc_binary()
        if opusenc_binary is None:
            raise SileroEngineError(
                "response_format 'opus' requires 'opusenc' to be installed and available in PATH.",
                'response_format',
                400,
            )
        return opusenc_binary

    def _fallback_speed_wav(self, audio: np.ndarray, speed: float, sample_rate: int) -> bytes:
        logger.warning('sox is unavailable or failed for tempo; using fallback speed processing.')
        fallback_audio = self._resample_for_speed(audio, speed)
        fallback_pcm = (fallback_audio * 32767.0).astype(np.int16).tobytes()
        return self._serialize_wav(fallback_pcm, sample_rate)

    def _apply_speed_to_wav(
        self, wav_audio: bytes, speed: float, audio: np.ndarray, sample_rate: int
    ) -> bytes:
        if speed == 1.0:
            return wav_audio

        sox_binary = self._find_sox_binary()
        if sox_binary is None:
            return self._fallback_speed_wav(audio=audio, speed=speed, sample_rate=sample_rate)

        with tempfile.TemporaryDirectory(prefix='silero-tts-') as temp_dir:
            source_path = os.path.join(temp_dir, 'input.wav')
            target_path = os.path.join(temp_dir, 'output.wav')

            with open(source_path, 'wb') as source_file:
                source_file.write(wav_audio)

            command = [sox_binary, source_path, target_path, 'tempo', f'{speed:g}']
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode != 0:
                logger.warning(
                    'sox tempo failed, using fallback speed processing: %s', process.stderr.strip()
                )
                return self._fallback_speed_wav(audio=audio, speed=speed, sample_rate=sample_rate)

            with open(target_path, 'rb') as target_file:
                return target_file.read()

    @staticmethod
    def _convert_with_sox(
        wav_audio: bytes, response_format: str, bitrate: str | None, sox_binary: str
    ) -> bytes:
        with tempfile.TemporaryDirectory(prefix='silero-tts-') as temp_dir:
            source_path = os.path.join(temp_dir, 'input.wav')
            target_path = os.path.join(temp_dir, f'output.{response_format}')

            with open(source_path, 'wb') as source_file:
                source_file.write(wav_audio)

            command = [sox_binary, source_path]
            if bitrate and response_format in {'mp3', 'opus'}:
                command.extend(['-C', bitrate])
            command.append(target_path)

            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode != 0:
                stderr_message = process.stderr.strip() or 'sox conversion failed.'
                raise SileroEngineError(stderr_message, 'response_format', 400)

            with open(target_path, 'rb') as target_file:
                return target_file.read()

    @staticmethod
    def _convert_with_opusenc(wav_audio: bytes, bitrate: str | None, opusenc_binary: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix='silero-tts-') as temp_dir:
            source_path = os.path.join(temp_dir, 'input.wav')
            target_path = os.path.join(temp_dir, 'output.opus')

            with open(source_path, 'wb') as source_file:
                source_file.write(wav_audio)

            command = [opusenc_binary, '--quiet']
            if bitrate is not None:
                command.extend(['--bitrate', bitrate])
            command.extend([source_path, target_path])

            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode != 0:
                stderr_message = process.stderr.strip() or 'opusenc conversion failed.'
                raise SileroEngineError(stderr_message, 'response_format', 400)

            with open(target_path, 'rb') as target_file:
                return target_file.read()

    @staticmethod
    def _normalize_bitrate(response_format: str, bitrate: object | None) -> str | None:
        if response_format not in {'mp3', 'opus'}:
            return None

        if bitrate is None:
            return None

        if isinstance(bitrate, str):
            normalized = bitrate.strip().lower()
            if not normalized:
                return None
            normalized = normalized.removesuffix('kbps').removesuffix('k').strip()
            if not normalized.isdigit():
                raise SileroEngineError(
                    "Parameter 'bitrate' must be one of: 64, 96, 128, 192.",
                    'bitrate',
                    400,
                )
            bitrate_value = int(normalized)
        elif isinstance(bitrate, int | float):
            if isinstance(bitrate, float) and not bitrate.is_integer():
                raise SileroEngineError(
                    "Parameter 'bitrate' must be one of: 64, 96, 128, 192.",
                    'bitrate',
                    400,
                )
            bitrate_value = int(bitrate)

        else:
            raise SileroEngineError(
                "Parameter 'bitrate' must be one of: 64, 96, 128, 192.",
                'bitrate',
                400,
            )

        if bitrate_value not in SUPPORTED_BITRATES:
            raise SileroEngineError(
                "Parameter 'bitrate' must be one of: 64, 96, 128, 192.",
                'bitrate',
                400,
            )

        return str(bitrate_value)

    def synthesize(
        self,
        model_name: str,
        voice: str,
        text: str,
        response_format: str,
        sample_rate: int,
        speed: float,
        bitrate: object | None = None,
    ) -> bytes:
        normalized_text = normalize_tts_text(text)
        if not normalized_text:
            raise SileroEngineError("Parameter 'input' must be a non-empty string.", 'input', 422)

        mapping = self._resolve_model_mapping(model_name)
        resolved_voice = self.resolve_voice(mapping.language, voice)

        if response_format not in OPENAI_RESPONSE_FORMATS:
            raise SileroEngineError(
                'Unsupported response_format. Use one of: wav, mp3, opus, flac, pcm.',
                'response_format',
                400,
            )

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise SileroEngineError(
                'Unsupported sample_rate. Use one of: 8000, 24000, 48000.',
                'sample_rate',
                400,
            )

        if speed < MIN_SPEED or speed > MAX_SPEED:
            raise SileroEngineError("Parameter 'speed' must be between 0.25 and 4.0.", 'speed', 400)

        model = self._load_model(mapping)
        with torch.no_grad():
            audio_tensor = model.apply_tts(
                text=normalized_text,
                speaker=resolved_voice,
                sample_rate=sample_rate,
            )

        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor)

        audio = audio_tensor.detach().cpu().float().numpy().squeeze()
        if audio.ndim == 0:
            audio = np.array([float(audio)], dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        if response_format == 'pcm':
            audio = self._resample_for_speed(audio, speed)
            audio_pcm = bytes((audio * 32767.0).astype(np.int16).tobytes())
            return audio_pcm

        audio_pcm = (audio * 32767.0).astype(np.int16).tobytes()
        wav_audio = self._serialize_wav(audio_pcm, sample_rate)
        wav_audio = self._apply_speed_to_wav(
            wav_audio=wav_audio,
            speed=speed,
            audio=audio,
            sample_rate=sample_rate,
        )

        if response_format == 'wav':
            return wav_audio

        bitrate_value = self._normalize_bitrate(response_format=response_format, bitrate=bitrate)

        if response_format == 'opus':
            opusenc_binary = self._require_opusenc_binary()
            return self._convert_with_opusenc(
                wav_audio=wav_audio,
                bitrate=bitrate_value,
                opusenc_binary=opusenc_binary,
            )

        sox_binary = self._require_sox_binary(response_format)
        return self._convert_with_sox(
            wav_audio=wav_audio,
            response_format=response_format,
            bitrate=bitrate_value,
            sox_binary=sox_binary,
        )
