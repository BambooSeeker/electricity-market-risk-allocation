"""Download public archive URLs and verify the recorded original SHA-256."""
from pathlib import Path
import csv
import hashlib
import urllib.request


def main():
    root = Path(__file__).resolve().parent
    with (root / 'docs/nyiso_source_manifest.csv').open(encoding='utf-8') as stream:
        for item in csv.DictReader(stream):
            path = root / item['file']
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                urllib.request.urlretrieve(item['source_url'], path)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != item['sha256']:
                raise ValueError(f'Archive differs from the recorded version: {path.name}')
            print('Verified', path.name)


if __name__ == '__main__':
    main()
