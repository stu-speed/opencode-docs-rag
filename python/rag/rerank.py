"""
Local cross-encoder reranker (ONNX Runtime + fast tokenizer, CPU).

The reranker is a Hugging Face repository that ships an ONNX export and a
fast tokenizer:

    <repo>/onnx/model.onnx
    <repo>/tokenizer.json

Files are cached under <data_dir>/models/<slug>/; missing files are
downloaded with huggingface_hub on first use. onnxruntime, tokenizers, and
huggingface_hub are already part of the engine environment (no new
dependencies). If a download fails and the files were not placed manually,
a clear error is raised and the query falls back to fusion order.
"""
import os
import shutil

import numpy as np

from .config import data_dir, rerank_max_tokens, rerank_model

# local filename -> path inside the HF repo
_REPO_FILES = {
    "model.onnx": "onnx/model.onnx",
    "tokenizer.json": "tokenizer.json",
}


def model_dir(model):
    """Local cache directory for a reranker model repo id."""
    slug = model.strip().strip("/").replace("/", "-")
    return os.path.join(str(data_dir()), "models", slug)


def ensure_model_files(model):
    """Make sure the reranker files exist locally; download any that are
    missing. Returns the model directory. Raises RuntimeError with a clear
    message when the files cannot be obtained."""
    dest = model_dir(model)
    os.makedirs(dest, exist_ok=True)
    missing = [f for f in _REPO_FILES if not os.path.exists(os.path.join(dest, f))]
    if not missing:
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            f"Reranker model files are missing from {dest} ({', '.join(missing)}) "
            "and huggingface_hub is not installed, so they cannot be downloaded. "
            "Install huggingface_hub or place the files manually."
        ) from e
    for f in missing:
        try:
            src = hf_hub_download(repo_id=model, filename=_REPO_FILES[f])
        except Exception as e:
            raise RuntimeError(
                f"Could not download '{_REPO_FILES[f]}' from '{model}' ({e}). "
                f"Place it manually at {os.path.join(dest, f)}."
            ) from e
        target = os.path.join(dest, f)
        tmp = f"{target}.part"
        # hf_hub_download may return a relative symlink into the HF cache;
        # copy the real file so the model dir is self-contained.
        shutil.copyfile(os.path.realpath(src), tmp)
        os.replace(tmp, target)
    return dest


class LocalOnnxReranker:
    """Cross-encoder scoring (query, chunk) pairs via ONNX Runtime (CPU)."""

    def __init__(self, model=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model = model or rerank_model()
        if not model:
            raise RuntimeError("Rerank is enabled but rerank.model is not set.")
        dest = ensure_model_files(model)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # errors only
        n_threads = os.cpu_count() or 1
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = n_threads
        self._session = ort.InferenceSession(
            os.path.join(dest, "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = [i.name for i in self._session.get_inputs()]
        self._tokenizer = Tokenizer.from_file(os.path.join(dest, "tokenizer.json"))
        # Truncate the longer sequence of the pair (the passage) first.
        self._tokenizer.enable_truncation(
            max_length=rerank_max_tokens(), strategy="longest_first"
        )

    def rerank(self, query, documents):
        """Score (query, doc) pairs. Returns one float per document, in the
        given order; higher = more relevant."""
        if not documents:
            return []
        encodings = self._tokenizer.encode_batch(
            [(query, doc) for doc in documents]
        )
        # Pad every pair to the batch's max length (onnxruntime needs a
        # rectangular input).
        max_len = max(len(e) for e in encodings)
        for e in encodings:
            e.pad(max_len)
        attr_map = {
            "input_ids": [e.ids for e in encodings],
            "attention_mask": [e.attention_mask for e in encodings],
            "token_type_ids": [e.type_ids for e in encodings],
        }
        # Feed only the inputs this export declares (onnxruntime rejects extras).
        feeds = {
            name: np.asarray(attr_map[name], dtype=np.int64)
            for name in self._inputs
            if name in attr_map
        }
        outputs = self._session.run(None, feeds)
        scores = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        return [float(x) for x in scores]
