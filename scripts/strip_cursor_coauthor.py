#!/usr/bin/env python3
"""Git msg-filter helper: remove Cursor co-author trailers from commit messages."""

import sys

for line in sys.stdin:
    if "Co-authored-by: Cursor" not in line:
        sys.stdout.write(line)
