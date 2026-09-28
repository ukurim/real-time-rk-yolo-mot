# Pinned upstream source archive

Repository: https://github.com/NirAharon/BoT-SORT

Commit: `251985436d6712aaf682aaaf5f71edb4987224bd`

License: MIT, Copyright (c) 2022 Nir Aharon; verbatim copy in `LICENSE`.

`upstream/*.py` are unmodified copies of `tracker/*.py` at that commit. They
are included for provenance and review, not imported by the application.
The C++ adaptation is `../../src/bot_sort.cpp`; configuration and differences
are documented in `../../docs/BOTSORT_PROVENANCE.md`.

SHA-256 checksums:

```text
12a15474ca15cec568da311e6cd89e593e3cd2bb5b790e75b90e1608306a3a5f  LICENSE
a0f717a5059ebfa626a63cea5b040a649acd7720072f503cabfa24d749f082f6  upstream/basetrack.py
ba6dd32ad4b96ee49d6f70654159269bd45ce15fd0fe5007f3a1b81bc257edf0  upstream/bot_sort.py
508c3a4ea0e9a72478ce2d1d29aa1eb8892b3804c76b524a16a34dca23dba9ba  upstream/gmc.py
c1a4c0acc71b9a91c0cdbe7b689f88712d68e7e8be4d7e54fb437f33fdbc9e5a  upstream/kalman_filter.py
020b2215c20467b31f7d8a977a2f557b8d1838f8baa5a5be54c0b8d14bbee4e1  upstream/matching.py
```

`generate_kalman_fixture.py` is an optional local verification helper. It
executes the archived Kalman implementation to regenerate the numerical fixture
used by the C++ tests. NumPy/SciPy are needed only for this helper.
