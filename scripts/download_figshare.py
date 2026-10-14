"""Download the three breast files from Figshare record 5330593 v2 and verify MD5."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
FILES=[('brca-rsem-fpkm-tcga-t.txt.gz',9150571,'f0c195f686a02cb6b757d9e553895ab4'),
       ('brca-rsem-fpkm-tcga.txt.gz',9150574,'47c9715d28e8fce38f9bbc558a193dfe'),
       ('breast-rsem-fpkm-gtex.txt.gz',9150577,'e2e7b608a21af5bfe8e187b0b8ddf5d1')]

def md5(path):
    digest=hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outdir',type=Path,default=Path('raw/figshare_5330593_v2'))
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    for name,file_id,expected in FILES:
        path=args.outdir/name
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(f'https://ndownloader.figshare.com/files/{file_id}',timeout=120) as source,temporary.open('wb') as target:
                while chunk:=source.read(1024*1024):target.write(chunk)
            if md5(temporary)!=expected:raise ValueError(f'Checksum mismatch: {temporary}')
            temporary.replace(path)
        if md5(path)!=expected:raise ValueError(f'Checksum mismatch: {path}')
        print(f'OK {name} md5={expected}')
    print('Normalized expression, NOT raw counts. See docs/DATA.md before transformation.')

if __name__=='__main__':main()
