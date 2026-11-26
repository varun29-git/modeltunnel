# Token-set launch history, 2026-10-01

This is a sanitized historical launch record. Instance IDs, process IDs, VM/resource names, local operator paths, and private bucket names are omitted. For the completed training outcomes, see `docs/RESULTS.md`.

## AWS preparation run

An earlier oversized Azure launch was stopped and its temporary work volume was not used as an output set. The authorized AWS preparation run then used an in-region `i4i.2xlarge` instance with 8 vCPUs, 64 GiB memory, and local NVMe. Source and destination remained in the same region; corpus concurrency was `SOURCE_WORKERS=6` and tokenization used `WORKERS=7`. No token set was claimed complete until pack validation and nested manifest generation finished.

## Aborted launch and data-cap correction

- The aborted worker used `Standard_D96ads_v7` with 96 vCPUs and 377 GiB RAM, plus a temporary RAID0 work volume.
- The destination was an authorized private prefix represented here as `s3://your-research-bucket/data/modeltunnelv2_tokensets_20261001/`.
- Tokenizer: frozen `tokenizer/tokenizer-65k.json`, SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`.
- Training mix: English 50%, math 10%, code 10%, seven Indic languages 30%.
- Training-document shuffle: seeded BLAKE2b-128 bucket sort after split and deduplication, seed `terminus-tokensets-20261001-v1`. Validation/test document order remained stable. The 3B and 6B manifests were nested in 10B.
- The aborted attempt reached concurrent source builders but not the planned tokenizer stage.

That attempt revealed a capacity error: source `max_bytes` measures compressed Parquet bytes, while the old inventory estimate multiplied those bytes by fertility measured on uncompressed text. After six ClimbMix objects, the cursor reported 1.51 GB of accepted text; `shard_00000.parquet` alone is about 92 MB compressed. The original 30 GiB source cap would have overbuilt the English share.

The resumed build retained raw-byte upper bounds but stopped on accepted decoded UTF-8 text: 25 GB ClimbMix; 2 GB per math source; 3 GB code concepts and 1.5 GB scientific coding; 4 GB verified Sangraha plus 1.5 GB FineWeb2 per Indic language. These were capacity targets with headroom, not token-count claims; the pack gate checked actual per-stream token counts.

Objects were selected in fixed hash order rather than only by first S3 key. Within an object, accepted records were a prefix of row order; document shuffling did not make that prefix a uniform source sample. Source reports recorded partial final objects.

Validation before launch: 26 focused data/shuffle tests plus Python and shell compile checks passed. A local PyTorch 2.2 environment could not run the broader model suite because `enable_gqa` required a newer build. No GPU training was started by this token-set preparation record.
