"""Bounded, read-only S3 access for corpus inventory and ingestion.

Everything here is read-only: prefixes are listed, objects are fetched in
bounded byte ranges or streamed, and nothing is ever written back to S3.
Credentials come from the ambient IAM environment (aws CLI/boto3 defaults);
no credentials are stored in this repository, manifests, or receipts.

`S3RangeReader` is a seekable file-like view over one S3 object that fetches
only the byte ranges actually read (with a small read-ahead cache), which
makes Parquet footer/schema inspection cheap. `LocalCache` bounds how much
object data is kept on disk during streaming ingestion.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import boto3


def s3_client():
    return boto3.client("s3")


def split_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI needs bucket and key: {uri}")
    return bucket, key


def list_prefix(uri, *, client=None, max_keys=None):
    """List one prefix. Returns [{"key", "size", "etag"}...] sorted by key."""
    client = client or s3_client()
    bucket, prefix = split_s3_uri(uri)
    objects = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for entry in page.get("Contents", []):
            objects.append({"key": entry["Key"], "size": entry["Size"],
                            "etag": entry["ETag"].strip('"')})
            if max_keys and len(objects) >= max_keys:
                return sorted(objects, key=lambda o: o["key"])
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return sorted(objects, key=lambda o: o["key"])


class S3RangeReader:
    """Seekable read-only view over one S3 object with range caching."""

    def __init__(self, uri, *, client=None, cache_bytes=8 * 1024 * 1024):
        self.client = client or s3_client()
        self.bucket, self.key = split_s3_uri(uri)
        head = self.client.head_object(Bucket=self.bucket, Key=self.key)
        self.size = head["ContentLength"]
        self.etag = head["ETag"].strip('"')
        self.cache_bytes = cache_bytes
        self._cache_start = 0
        self._cache = b""
        self._pos = 0
        self.closed = False

    # file-object protocol -------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self._pos

    def seek(self, position, whence=0):
        if whence == 0:
            self._pos = position
        elif whence == 1:
            self._pos += position
        elif whence == 2:
            self._pos = self.size + position
        else:
            raise ValueError("invalid whence")
        if not 0 <= self._pos <= self.size:
            raise ValueError("seek out of range")
        return self._pos

    def _fetch(self, start, length):
        end = min(self.size, start + max(length, self.cache_bytes)) - 1
        body = self.client.get_object(Bucket=self.bucket, Key=self.key,
                                      Range=f"bytes={start}-{end}")["Body"].read()
        self._cache_start = start
        self._cache = body

    def read(self, length=-1):
        if self.closed:
            raise ValueError("read on closed S3RangeReader")
        if length is None or length < 0:
            length = self.size - self._pos
        length = min(length, self.size - self._pos)
        if length == 0:
            return b""
        if not (self._cache_start <= self._pos
                and self._pos + length <= self._cache_start + len(self._cache)):
            self._fetch(self._pos, length)
        offset = self._pos - self._cache_start
        data = self._cache[offset:offset + length]
        self._pos += len(data)
        return data

    def close(self):
        self.closed = True
        self._cache = b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LocalCache:
    """Bounded on-disk cache for streamed objects.

    Files are stored under root/sha1(key); when total bytes exceed
    `capacity_bytes`, the least-recently-used files are evicted. A JSON index
    makes restarts reuse what is already downloaded.
    """

    def __init__(self, root, *, capacity_bytes=4 * 1024**3):
        self.root = Path(root)
        self.capacity_bytes = capacity_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "cache_index.json"
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text())
        else:
            self.index = {}

    def _target(self, key):
        return self.root / hashlib.sha1(key.encode()).hexdigest()

    def _evict(self, protect=None):
        total = sum(entry["size"] for entry in self.index.values())
        if total <= self.capacity_bytes:
            return
        for key, entry in sorted(self.index.items(), key=lambda kv: kv[1]["atime"]):
            if total <= self.capacity_bytes:
                break
            if key == protect:
                continue  # never evict the file a caller is about to read
            path = self._target(key)
            if path.exists():
                path.unlink()
            total -= entry["size"]
            del self.index[key]

    def fetch(self, uri, *, client=None, expected_size=None, expected_etag=None):
        """Download an object into the cache (or reuse it) and return its path."""
        client = client or s3_client()
        bucket, key = split_s3_uri(uri)
        entry = self.index.get(key)
        if entry is not None and self._target(key).exists():
            if expected_size is None or entry["size"] == expected_size:
                if expected_etag is None or entry.get("etag") == expected_etag:
                    entry["atime"] = time.time()
                    self._save()
                    return self._target(key)
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        target = self._target(key)
        digest = hashlib.sha256()
        size = 0
        with target.open("wb") as handle:
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_size is not None and size != expected_size:
            target.unlink()
            raise ValueError(f"downloaded size mismatch for {uri}")
        self.index[key] = {"size": size, "sha256": digest.hexdigest(),
                           "etag": response["ETag"].strip('"'), "atime": time.time()}
        self._evict(protect=key)
        self._save()
        return target

    def _save(self):
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.index, indent=1, sort_keys=True))
        os.replace(temporary, self.index_path)
