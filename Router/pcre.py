"""Minimal shim mapping the optional `pcre` dependency onto stdlib `re`.

gptqmodel.utils.logger only uses pcre.compile for an ANSI-escape regex, so a
thin pass-through to the standard library is sufficient for inference use.
"""
import re as _re

compile = _re.compile
match = _re.match
search = _re.search
sub = _re.sub
findall = _re.findall
split = _re.split
escape = _re.escape

I = _re.I
IGNORECASE = _re.IGNORECASE
M = _re.M
MULTILINE = _re.MULTILINE
S = _re.S
DOTALL = _re.DOTALL

error = _re.error
Pattern = _re.Pattern
