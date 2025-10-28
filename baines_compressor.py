#!/usr/bin/env python3
"""
Legacy launcher retained for backward compatibility. Delegates to ``baines.cli``.
"""

from __future__ import annotations

from baines.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
