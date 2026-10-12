"""Ollama access, and the embedders.

Ollama serves the summarizer (DeepSeek-R1) and the default embedder
(nomic-embed-text). It will not embed with a generative model, so the DeepSeek
embedder pools the model's hidden states through transformers instead.
"""

import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

SUMMARY_MODEL = "deepseek-r1:1.5b"
EMBED_MODEL = "nomic-embed-text"
DEEPSEEK_HF_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Texts at most this long, on one line, are their own summary: asking a model
# to shorten a label only paraphrases it.
SUMMARY_PASSTHROUGH = 80


class Ollama:
    def __init__(self, url=OLLAMA_URL):
        self.url = url.rstrip("/")

    def alive(self):
        try:
            return requests.get(f"{self.url}/api/version", timeout=2).ok
        except requests.RequestException:
            return False

    def ensure_server(self, progress=print):
        if self.alive():
            return
        exe = shutil.which("ollama")
        if exe is None:
            raise RuntimeError("Ollama is not installed: `brew install ollama`, or see https://ollama.com")
        progress("starting ollama serve")
        subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(60):
            time.sleep(0.5)
            if self.alive():
                return
        raise RuntimeError(f"ollama serve did not come up at {self.url}")

    def has_model(self, name):
        tags = requests.get(f"{self.url}/api/tags", timeout=10).json().get("models", [])
        names = {m["name"] for m in tags}
        return name in names or f"{name}:latest" in names

    def ensure_model(self, name, progress=print):
        if self.has_model(name):
            return
        progress(f"pulling {name}")
        with requests.post(f"{self.url}/api/pull", json={"model": name}, stream=True, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise RuntimeError(f"pulling {name}: {msg['error']}")
                if msg.get("total"):
                    pct = 100.0 * msg.get("completed", 0) / msg["total"]
                    progress(f"pulling {name}: {msg.get('status', '')} {pct:.0f}%")
                else:
                    progress(f"pulling {name}: {msg.get('status', '')}")

    def embed(self, model, texts):
        r = requests.post(f"{self.url}/api/embed", json={"model": model, "input": list(texts)}, timeout=600)
        r.raise_for_status()
        return np.asarray(r.json()["embeddings"], dtype=np.float32)

    def generate(self, model, prompt):
        # think=False asks a reasoning model to skip its <think> block; the
        # regex below strips one anyway, for servers that ignore the flag.
        body = {"model": model, "prompt": prompt, "stream": False, "think": False, "options": {"temperature": 0.2}}
        r = requests.post(f"{self.url}/api/generate", json=body, timeout=600)
        r.raise_for_status()
        return r.json()["response"]


def one_line(text, limit=100):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    lines = [ln.strip().strip('"').strip() for ln in text.strip().splitlines() if ln.strip()]
    line = lines[0] if lines else ""
    line = re.sub(r"^(summary|one[- ]line summary)\s*:\s*", "", line, flags=re.I)
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


class Summarizer:
    def __init__(self, ollama, model=SUMMARY_MODEL):
        self.ollama = ollama
        self.model = model

    def prepare(self, progress=print):
        self.ollama.ensure_model(self.model, progress)

    def __call__(self, text):
        text = text.strip()
        if len(text) <= SUMMARY_PASSTHROUGH and "\n" not in text:
            return text
        prompt = (
            "Summarize the following text in one line of at most 12 words. "
            "Reply with the summary only.\n\n" + text
        )
        return one_line(self.ollama.generate(self.model, prompt)) or one_line(text)


class OllamaEmbedder:
    """Embedding through an Ollama embedding model."""

    def __init__(self, ollama, model=EMBED_MODEL):
        self.ollama = ollama
        self.model = model
        self.name = f"ollama:{model}"

    def prepare(self, progress=print):
        self.ollama.ensure_model(self.model, progress)

    def __call__(self, texts):
        # nomic-embed-text is trained with task prefixes; "clustering" is the
        # one meant for laying texts out by topic.
        if self.model.startswith("nomic-embed-text"):
            texts = [f"clustering: {t}" for t in texts]
        return self.ollama.embed(self.model, texts)


class DeepSeekEmbedder:
    """Mean-pooled last hidden states of a DeepSeek model, through transformers.

    A decoder is not trained to make these vectors comparable, so expect
    looser clusters than from a real embedding model.
    """

    def __init__(self, model=DEEPSEEK_HF_MODEL):
        self.model_name = model
        self.name = f"hf:{model}"
        self.tokenizer = None
        self.model = None

    def prepare(self, progress=print):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("--embedder deepseek needs `pip install -e .[hyp-deepseek]`") from e
        progress(f"loading {self.model_name} (downloads on first use)")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float32)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device).eval()

    def __call__(self, texts, batch=16):
        import torch

        out = []
        for i in range(0, len(texts), batch):
            enc = self.tokenizer(list(texts[i : i + batch]), padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
