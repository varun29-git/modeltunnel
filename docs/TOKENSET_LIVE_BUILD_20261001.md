# Token-set launch history, 2026-10-01

## Active AWS run

The Azure launch below was stopped after the owner pointed out that it was
oversized and in the wrong cloud. Azure `tok64` was **deallocated** before
merge/tokenization/upload. Its temporary work volume is not an output set.

- AWS EC2 `i-0be4f12ac9f0c1094`, `i4i.2xlarge`, `us-east-1a`: 8 vCPUs,
  64 GiB memory, and 1.875 TB local NVMe mounted at `/data`.
- In-region S3 source and destination: no cross-cloud source transfer.
- Corpus concurrency: `SOURCE_WORKERS=6`; tokenization: `WORKERS=7`.
- Pipeline PID 27339 started about 05:55 UTC. Log: `/data/tokenization.log`.
- Frozen tokenizer SHA, shuffle seed, training mix, source text-volume caps,
  and S3 destination are the same as below.
- No completed token sets were claimed at launch; upload follows the 10B
  pack/validation gate and nested 3B/6B manifest generation.

## Aborted Azure launch and data-cap correction

- VM: Azure `H100_GROUP/tok64`, `Standard_D96ads_v7` (96 vCPUs, 377 GiB RAM).
- Work volume: RAID0 across six temporary NVMe disks, XFS at `/data` (5.2 TiB usable).
- Destination: `s3://quark-research-repository-storage/data/modeltunnelv2_tokensets_20261001/`.
- Tokenizer: frozen `tokenizer/tokenizer-65k.json`, SHA-256
  `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`.
- Training mix: English 50%, math 10%, code 10%, seven Indic languages 30%.
- Training-document shuffle: seeded BLAKE2b-128 bucket sort after split and
  deduplication, seed `terminus-tokensets-20261001-v1`. Validation and test
  document order is left stable. The 3B and 6B manifests are nested in 10B.
- Execution before deallocation: 19 concurrent source builders; the planned
  tokenizer stage (88 workers) was never reached. The revised Azure job had
  launched as PID 5480 at approximately 05:37 UTC.

The first attempt was stopped during corpus preparation and its incomplete
derived output retained at `/data/work_aborted_20261001_0528`. It revealed a
capacity error: source `max_bytes` measures compressed Parquet bytes, while
the old inventory estimate multiplied those bytes by fertility measured on
uncompressed text. After six ClimbMix objects, the live cursor reported
1.51 GB of accepted text; the object `shard_00000.parquet` alone is about
92 MB compressed. The original 30 GiB source cap would overbuild the English
share considerably.

The resumed build keeps raw-byte limits as upper bounds but stops on accepted
decoded UTF-8 text: 25 GB ClimbMix; 2 GB per math source; 3 GB code concepts
and 1.5 GB scientific coding; 4 GB verified Sangraha plus 1.5 GB FineWeb2
per Indic language. These targets include headroom based on the bounded
tokenizer sample. The final pack gate must verify the **actual** token count
per stream; these text-volume estimates are not a claim that all three sets
already exist.

Objects are selected by a fixed hash order to avoid taking only the first
S3 keys. Within an object, accepted records are a prefix of its row order;
document shuffling makes the training order reproducible but does not turn
that prefix into a uniform sample of the source. Source reports record any
partial final object.

Validation before launch: 26 focused data/shuffle tests passed; Python and
shell compile checks passed. The broader CPU model test suite has an
unrelated local PyTorch 2.2 limitation (`enable_gqa` requires a newer build).
No GPU training was started.
