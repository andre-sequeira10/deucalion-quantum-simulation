#!/usr/bin/env python3
"""Offline integrity and syntax checks; never imports or executes experiments."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    for name, record in manifest.items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise RuntimeError(f'Source hash mismatch: {name}')
    for path in (root / 'experiments').rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for folder in ('experiments', 'recorded-results'):
        for path in (root / folder).rglob('*.json'):
            json.loads(path.read_text())
    print(f'PASS: {len(manifest)} copied-file hashes; Python and JSON syntax.')


if __name__ == '__main__':
    main()
