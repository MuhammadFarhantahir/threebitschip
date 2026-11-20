"""Fetch analysis inputs from public archives into the ignored data directory."""
import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
MICA_ARCHIVE = 'mica_mics/micapipe_MICs_v1.1.zip'


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(asset):
    path = ROOT / 'data' / asset['path']
    if path.is_file() and checksum(path) == asset['sha256']:
        print(f'Existing verified: {asset["path"]}', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.part', delete=False) as stream:
        temp = Path(stream.name)
    try:
        for attempt in range(3):
            try:
                with urlopen(asset['url'], timeout=45) as response, temp.open('wb') as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                if checksum(temp) != asset['sha256']:
                    raise ValueError(f'Archive returned unexpected bytes for {asset["path"]}')
                temp.replace(path)
                print(f'Downloaded: {asset["path"]}', flush=True)
                return
            except OSError:
                if attempt == 2:
                    raise
    finally:
        temp.unlink(missing_ok=True)


def extract_mica():
    directory = ROOT / 'data' / 'mica_mics'
    with ZipFile(ROOT / 'data' / MICA_ARCHIVE) as archive:
        for number in range(1, 51):
            subject = f'sub-HC{number:03d}'
            session = f'micapipe/{subject}/ses-01'
            prefix = f'{subject}_ses-01'
            archive.extract(f'{session}/dwi/{prefix}_space-dwinative_atlas-schaefer100_desc-sc.txt', directory)
            archive.extract(f'{session}/func/{prefix}_space-fsnative_atlas-schaefer100_desc-timeseries.txt', directory)


def main():
    datasets = json.loads((ROOT / 'dataset-assets.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=sorted(datasets), nargs='+')
    parser.add_argument('--list', action='store_true', help='Show source URLs without downloading')
    args = parser.parse_args()
    assets = [asset for dataset in dict.fromkeys(args.dataset) for asset in datasets[dataset]]
    if args.list:
        for asset in assets:
            print(f'{asset["path"]}: {asset["url"]}')
        return
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(fetch, assets):
            pass
    if 'mica' in args.dataset:
        extract_mica()


if __name__ == '__main__':
    main()
