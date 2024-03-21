import os
import gdown
import uvicorn
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Body,
    BackgroundTasks,
    UploadFile,
    File,
)
from pydantic import BaseModel
import torch
from transformers import pipeline
from pathlib import Path
from .diarization_pipeline import diarize
import requests

hf_token = os.environ.get(
    "HF_TOKEN",
)

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device="cuda:0",
)

app = FastAPI()


class WebhookBody(BaseModel):
    url: str
    header: dict[str, str] = {}


def process(
    url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    diarise_audio: bool,
    webhook: WebhookBody | None = None,
):
    errorMessage: str | None = None
    outputs = {}

    try:
        generate_kwargs = {
            "task": task,
            "language": None if language == "None" else language,
        }

        outputs = pipe(
            url,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if timestamp == "word" else True,
        )

        if diarise_audio is True:
            speakers_transcript = diarize(
                hf_token,
                url,
                outputs,
            )
            outputs["speakers"] = speakers_transcript
    except Exception as e:
        errorMessage = str(e)

    if webhook is not None:
        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(
                {"output": outputs, "status": "completed"}
                if errorMessage is None
                else {"error": errorMessage, "status": "error"}
            ),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs


@app.post("/")
def root(
    background_tasks: BackgroundTasks,
    x_admin_api_key=Header(),
    url: str = Body(),
    task: str = Body(default="transcribe", enum=["transcribe", "translate"]),
    language: str = Body(default="None"),
    batch_size: int = Body(default=64),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
    diarise_audio: bool = Body(
        default=False,
    ),
    webhook: WebhookBody | None = None,
    is_async: bool = Body(default=False),
):
    if admin_key is not None:
        if x_admin_api_key != admin_key:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    if url.lower().startswith("http") is False:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if diarise_audio is True and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if is_async is True and webhook is None:
        raise HTTPException(
            status_code=400, detail="Webhook is required for async tasks"
        )

    try:
        if is_async is True:
            background_tasks.add_task(
                process,
                url,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook,
            )
            return {
                "message": "Task is being processed in the background",
                "status": "processing",
            }
        else:
            outputs = process(
                url,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook,
            )
        return {"output": outputs, "status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = "whisper-1",
    language: str = None,
    prompt: str = None,
    response_format: str = "json",
    temperature: float = 0.0,
    timestamp_granularities: list = ["segment"],
):

    if response_format not in ["json", "text", "srt", "verbose_json", "vtt"]:
        raise HTTPException(
            status_code=400, detail="Invalid response format specified."
        )

    if response_format == "verbose_json" and "word" not in timestamp_granularities:
        raise HTTPException(
            status_code=400, detail="Word-level timestamps are not supported."
        )

    try:
        url = f"https://drive.google.com/uc?id={prompt}"
        file = gdown.download(url, "t.wav")

        # if not os.path.isfile(audio_path):
        #     raise HTTPException(status_code=400, detail="Audio file not found.")

        with open("t.wav", "rb") as audio_file:
            audio_bytes = audio_file.read()

        result = pipe(
            audio_bytes,
            chunk_length_s=30,
        )

        if response_format == "json":
            return result
        elif response_format == "text":
            return result["text"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
