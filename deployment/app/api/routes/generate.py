from __future__ import annotations

import base64
import inspect
import logging
import time
from typing import Any, AsyncIterator

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from numpy.typing import NDArray

from app.api.deps import get_server
from app.core.metrics import (
    GENERATE_AUDIO_SECONDS_TOTAL,
    GENERATE_STREAM_BYTES_TOTAL,
    GENERATE_TTFB_SECONDS,
)
from app.schemas.http import ErrorResponse, GenerateRequest
from app.services.mp3 import stream_mp3, stream_pcm

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generation"])


def _decode_latents_base64(value: str, field_name: str, feat_dim: int) -> bytes:
    try:
        latents = base64.b64decode(value)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 in {field_name}: {e}") from e

    try:
        np.frombuffer(latents, dtype=np.float32).reshape(-1, feat_dim)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid latent payload in {field_name}: {e}") from e

    return latents


def _validate_generate_prompt(req: GenerateRequest) -> None:
    has_wav = req.prompt_wav_base64 is not None or req.prompt_wav_format is not None
    has_latents = req.prompt_latents_base64 is not None
    has_ref_wav = req.ref_audio_wav_base64 is not None or req.ref_audio_wav_format is not None
    has_ref_latents = req.ref_audio_latents_base64 is not None

    if has_wav and has_latents:
        raise HTTPException(
            status_code=400,
            detail="prompt_wav_* and prompt_latents_base64 are mutually exclusive",
        )

    if has_ref_wav and has_ref_latents:
        raise HTTPException(
            status_code=400,
            detail="ref_audio_wav_* and ref_audio_latents_base64 are mutually exclusive",
        )

    if has_ref_wav and (req.ref_audio_wav_base64 is None or req.ref_audio_wav_format is None):
        raise HTTPException(
            status_code=400,
            detail="reference wav requires ref_audio_wav_base64 + ref_audio_wav_format",
        )

    if has_wav:
        if req.prompt_wav_base64 is None or req.prompt_wav_format is None:
            raise HTTPException(
                status_code=400,
                detail="wav prompt requires prompt_wav_base64 + prompt_wav_format",
            )
        if req.prompt_text is None or req.prompt_text == "":
            raise HTTPException(status_code=400, detail="wav prompt requires non-empty prompt_text")
        return

    if has_latents:
        if req.prompt_text is None or req.prompt_text == "":
            raise HTTPException(status_code=400, detail="latents prompt requires non-empty prompt_text")
        return

    if req.prompt_text not in (None, ""):
        raise HTTPException(status_code=400, detail="prompt_text is not allowed for zero-shot")


@router.post(
    "/generate",
    response_class=StreamingResponse,
    summary="Generate audio (streaming MP3 or PCM)",
    responses={
        200: {
            "description": "Streamed audio bytes (MP3 by default, or raw s16le PCM when response_format='pcm')",
            "content": {
                "audio/mpeg": {
                    "schema": {"type": "string", "format": "binary"},
                },
                "audio/L16": {
                    "schema": {"type": "string", "format": "binary"},
                },
            },
            "headers": {
                "X-Audio-Sample-Rate": {
                    "description": "Audio sample rate in Hz.",
                    "schema": {"type": "integer"},
                },
                "X-Audio-Channels": {
                    "description": "Number of audio channels.",
                    "schema": {"type": "integer"},
                },
                "X-Audio-Encoding": {
                    "description": "Audio encoding of the stream ('mp3' or 's16le').",
                    "schema": {"type": "string"},
                },
            },
        },
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def generate(
    req: GenerateRequest,
    request: Request,
    server: Any = Depends(get_server),
) -> StreamingResponse:
    """Generate speech audio as a streamed byte stream.

    The output encoding is selected via ``response_format``: MP3 (``audio/mpeg``,
    default) or raw signed 16-bit little-endian mono PCM (``audio/L16``).
    The response is streamed and may terminate early if the client disconnects or
    an internal error occurs after streaming has started.
    """

    _validate_generate_prompt(req)

    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="server misconfigured: missing app.state.cfg")

    model_info = await server.get_model_info()
    sample_rate = int(model_info["sample_rate"])
    channels = int(model_info["channels"])
    feat_dim = int(model_info["feat_dim"])
    if channels != 1:
        raise HTTPException(status_code=500, detail=f"Only mono is supported (channels={channels})")

    if req.lora_name is not None:
        registered_loras = {str(item["name"]) for item in await server.list_loras()}
        if req.lora_name not in registered_loras:
            raise HTTPException(status_code=400, detail=f"LoRA '{req.lora_name}' is not registered")

    prompt_latents: bytes | None = None
    ref_audio_latents: bytes | None = None
    prompt_text = ""
    if req.prompt_wav_base64 is not None:
        try:
            wav = base64.b64decode(req.prompt_wav_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in prompt_wav_base64: {e}") from e
        assert req.prompt_wav_format is not None
        assert req.prompt_text is not None
        prompt_latents = await server.encode_latents(wav, req.prompt_wav_format)
        prompt_text = req.prompt_text
    elif req.prompt_latents_base64 is not None:
        prompt_latents = _decode_latents_base64(req.prompt_latents_base64, "prompt_latents_base64", feat_dim)
        assert req.prompt_text is not None
        prompt_text = req.prompt_text

    if req.ref_audio_wav_base64 is not None:
        try:
            wav = base64.b64decode(req.ref_audio_wav_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in ref_audio_wav_base64: {e}") from e
        assert req.ref_audio_wav_format is not None
        ref_audio_latents = await server.encode_latents(wav, req.ref_audio_wav_format)
    elif req.ref_audio_latents_base64 is not None:
        ref_audio_latents = _decode_latents_base64(req.ref_audio_latents_base64, "ref_audio_latents_base64", feat_dim)

    generate_kwargs = {
        "target_text": req.target_text,
        "prompt_latents": prompt_latents,
        "prompt_text": prompt_text,
        "max_generate_length": req.max_generate_length,
        "temperature": req.temperature,
        "cfg_value": req.cfg_value,
        "lora_name": req.lora_name,
    }

    generate_params = inspect.signature(server.generate).parameters

    if ref_audio_latents is not None:
        generate_kwargs["ref_audio_latents"] = ref_audio_latents

    if ref_audio_latents is not None:
        if "ref_audio_latents" not in generate_params:
            raise HTTPException(status_code=400, detail="Reference audio is not supported by the loaded model")

    if req.seed is not None:
        generate_kwargs["seed"] = req.seed

    if req.seed is not None:
        if "seed" not in generate_params:
            raise HTTPException(status_code=400, detail="Seed is not supported by the loaded model")

    # --- Diagnostic logging: request dimensions ---
    latents_len = len(prompt_latents) // 4 if prompt_latents else 0  # float32 count
    ref_latents_len = len(ref_audio_latents) // 4 if ref_audio_latents else 0
    logger.info(
        "/generate: target_text=%r (%d chars), prompt_text=%r (%d chars), "
        "latents=%d floats, ref_latents=%d floats, max_generate_length=%d, "
        "temperature=%.2f, cfg_value=%.2f, lora=%s, seed=%s, format=%s",
        req.target_text[:60],
        len(req.target_text),
        (prompt_text[:40] + "...") if len(prompt_text) > 40 else prompt_text,
        len(prompt_text),
        latents_len,
        ref_latents_len,
        req.max_generate_length,
        req.temperature,
        req.cfg_value,
        req.lora_name,
        req.seed,
        req.response_format,
    )

    stream = server.generate(**generate_kwargs)

    first_chunk: NDArray[np.float32] | None = None
    stream_exhausted = False
    try:
        first_chunk = await anext(stream)
    except StopAsyncIteration:
        stream_exhausted = True
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    start_t = time.perf_counter()
    ttfb_recorded = False
    chunk_count = 0
    total_bytes = 0
    gen_error: BaseException | None = None

    async def wav_chunks() -> AsyncIterator[NDArray[np.float32]]:
        nonlocal chunk_count
        if first_chunk is not None:
            chunk_count += 1
            GENERATE_AUDIO_SECONDS_TOTAL.inc(float(first_chunk.shape[0]) / float(sample_rate))
            logger.debug(
                "/generate wav_chunk #%d: shape=%s dtype=%s",
                chunk_count,
                first_chunk.shape,
                first_chunk.dtype,
            )
            yield first_chunk

        if stream_exhausted:
            return

        try:
            async for chunk in stream:
                chunk_count += 1
                GENERATE_AUDIO_SECONDS_TOTAL.inc(float(chunk.shape[0]) / float(sample_rate))
                logger.debug(
                    "/generate wav_chunk #%d: shape=%s dtype=%s",
                    chunk_count,
                    chunk.shape,
                    chunk.dtype,
                )
                yield chunk
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as exc:
            gen_error = exc  # noqa: F841 — read by body() for diagnostics
            logger.error(
                "/generate: server.generate() raised %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            raise

    if req.response_format == "pcm":
        audio_stream = stream_pcm(request=request, wav_chunks=wav_chunks())
        media_type = f"audio/L16;rate={sample_rate};channels={channels}"
        audio_encoding = "s16le"
    else:
        audio_stream = stream_mp3(
            request=request,
            wav_chunks=wav_chunks(),
            sample_rate=sample_rate,
            mp3=cfg.mp3,
        )
        media_type = "audio/mpeg"
        audio_encoding = "mp3"

    async def body() -> AsyncIterator[bytes]:
        nonlocal ttfb_recorded, total_bytes
        try:
            async for b in audio_stream:
                if not ttfb_recorded:
                    GENERATE_TTFB_SECONDS.observe(time.perf_counter() - start_t)
                    ttfb_recorded = True
                total_bytes += len(b)
                GENERATE_STREAM_BYTES_TOTAL.inc(len(b))
                yield b
        except Exception as exc:
            logger.error(
                "/generate body() error after %d bytes, %d wav chunks: %s: %s",
                total_bytes,
                chunk_count,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            # Don't re-raise — StreamingResponse will just close the connection.
            # The error is already logged.

        if not ttfb_recorded:
            GENERATE_TTFB_SECONDS.observe(time.perf_counter() - start_t)

        elapsed_ms = round((time.perf_counter() - start_t) * 1000)
        if total_bytes == 0:
            logger.warning(
                "/generate: EMPTY RESPONSE — 0 %s bytes after %dms, "
                "%d wav chunks, gen_error=%s. "
                "Likely cause: prompt_len + max_generate_length (%d) > max_model_len. "
                "Check container NANOVLLM_SERVERPOOL_MAX_MODEL_LEN.",
                audio_encoding,
                elapsed_ms,
                chunk_count,
                gen_error,
                req.max_generate_length,
            )
        else:
            logger.info(
                "/generate: streamed %d %s bytes in %dms (%d wav chunks)",
                total_bytes,
                audio_encoding,
                elapsed_ms,
                chunk_count,
            )

    return StreamingResponse(
        body(),
        media_type=media_type,
        headers={
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": str(channels),
            "X-Audio-Encoding": audio_encoding,
        },
    )
