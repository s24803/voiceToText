from transformers import pipeline
from datasets import load_dataset
from datasets import Audio
from evaluate import evaluator


minds = load_dataset("PolyAI/minds14", name="en-AU", split="train[:40]")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

facebook = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
distil = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")

task_evaluator = evaluator("automatic-speech-recognition")

task_evaluator.PIPELINE_KWARGS.pop('truncation', None)

f_results = task_evaluator.compute(
    model_or_pipeline=facebook,
    data=minds,
    input_column="audio",
    label_column="english_transcription",
    metric="wer"
)

w_results = task_evaluator.compute(
    model_or_pipeline=whisper,
    data=minds,
    input_column="audio",
    label_column="english_transcription",
    metric="wer"
)

d_results = task_evaluator.compute(
    model_or_pipeline=distil,
    data=minds,
    input_column="audio",
    label_column="english_transcription",
    metric="wer"
)

print(f_results)
print(w_results)
print(d_results)
