# Security and deployment

GitHub Pages serves static scripts, styles, and JSON. Public replay text is inserted as text, and outbound post links use an HTTPS hostname allowlist. The deployed pages declare a same-origin Content Security Policy.

`python serve.py` binds to loopback and serves an explicit file allowlist. It refuses symlinks, paths outside the site root, and write methods. Its HTTP headers apply only to the local preview.

The Colab downloads source files and the saved classifier from commit-pinned URLs and checks SHA-256 before use. FinTwitBERT uses a pinned revision, safetensors, and disabled remote code. The raw post archive, local weights, and environment files are excluded from version control. Run `python verify_submission.py` to exercise the input-integrity and preview-server tests.
