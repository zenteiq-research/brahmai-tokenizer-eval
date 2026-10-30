# Security

Report a security problem privately to research@zenteiq.ai, not in a public issue. Include the
version or commit, what you observed, and the steps to reproduce it.

This covers the code in this repository, for example `bte data install`, which unpacks an archive
downloaded from the internet, and the loading of tokenizer files. `--trust-remote-code` runs code
from a tokenizer's own repository by design; only use it for repositories you trust. Problems in a
tokenizer or dataset hosted elsewhere belong with its maintainers.
