from transformers import pipeline
from datasets import load_dataset
from datasets import Audio
from evaluate import evaluator, load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


minds = load_dataset("PolyAI/minds14", name="en-AU", split="train[:40]")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

example = minds[0]

facebook = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
distil = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")

f_output = facebook(example["audio"]["array"])
w_output = whisper(example["audio"]["array"])
d_output = distil(example["audio"]["array"])

correct = example["english_transcription"]

task_evaluator = evaluator("automatic-speech-recognition")
# wer_metric = load("wer")
#
# normalizer = BasicTextNormalizer()
# f_wer = wer_metric.compute(references=[normalizer(correct)], predictions=[normalizer(f_output['text'])])
# w_wer = wer_metric.compute(references=[normalizer(correct)], predictions=[normalizer(w_output['text'])])
# d_wer = wer_metric.compute(references=[normalizer(correct)], predictions=[normalizer(d_output['text'])])
#
# print(f_wer)
# print(w_wer)
# print(d_wer)

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
