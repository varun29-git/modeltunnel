"""Fetch the pinned RWKV-7 World byte vocabulary used by Terminus."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


UPSTREAM_COMMIT = "aeda8a06fc4803655473c7e7359b1d7166e2c604"
SHA256 = "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
URL = ("https://raw.githubusercontent.com/BlinkDL/RWKV-LM/" + UPSTREAM_COMMIT
       + "/RWKV-v7/rwkv_vocab_v20230424.txt")
DEFAULT_PATH = Path("data/tokenizer/rwkv-7-world/rwkv_vocab_v20230424.txt")


def fetch(path=DEFAULT_PATH):
    path = Path(path)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
            raise FileExistsError(f"existing tokenizer has the wrong SHA-256: {path}")
        return path
    request = urllib.request.Request(URL, headers={"User-Agent": "terminus-tokenizer-fetch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("downloaded RWKV tokenizer did not match the pinned SHA-256")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".rwkv_vocab_", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    if path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    print(fetch(parser.parse_args().output))
