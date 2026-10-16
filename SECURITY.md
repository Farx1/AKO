# Security policy

## Supported versions

Only the `main` branch is maintained. There are no released versions.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub private vulnerability reporting
(Security tab, "Report a vulnerability") on https://github.com/Farx1/AKO.
Expect a first answer within 14 days.

## Scope

AKO loads and runs kernels from its own registry and reads a JSON decision
cache from `AKO_CACHE_PATH` (default `~/.ako/kernel_cache.json`). Treat that
cache as trusted input: anyone who can write it can choose which registered
kernel runs. `ako.models.load_hf_model` downloads checkpoints through
`transformers`; the usual cautions about untrusted checkpoints apply.
