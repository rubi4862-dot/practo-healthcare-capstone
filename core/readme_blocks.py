"""
core/readme_blocks.py — scripts write their MEASURED numbers into README.md
between <!-- NAME:START --> and <!-- NAME:END --> markers, so the README never
contains a number that a script did not actually produce.
"""

import re

import config


def write_block(name: str, content: str) -> None:
    text = config.README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(<!-- {name}:END -->)", re.DOTALL)
    if not pattern.search(text):
        raise KeyError(f"README has no {name} block")
    text = pattern.sub(lambda m: m.group(1) + content.rstrip() + "\n" + m.group(2), text)
    config.README_PATH.write_text(text, encoding="utf-8")
