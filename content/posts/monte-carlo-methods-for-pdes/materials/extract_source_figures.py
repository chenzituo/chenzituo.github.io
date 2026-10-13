"""Render attributed figure excerpts from verified PDFs with Poppler.
Usage: python extract_source_figures.py /path/to/pdf-directory
PDF names: guiding.pdf, wods.pdf, neural-shapes.pdf, composition.pdf.
No full papers are copied into the website. Requires pdftoppm and Pillow.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from PIL import Image
root = Path(__file__).resolve().parent.parent
sources = {s['key']: s for s in json.loads((root/'materials/sources.json').read_text())['sources']}
figures = json.loads((root/'materials/figure-sources.json').read_text())['figures']
for f in figures:
    if f['type'] != 'source figure excerpt': continue
    pdf = Path(sys.argv[1]) / (f['source_key'] + '.pdf')
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == sources[f['source_key']]['sha256'], pdf
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp)/'page')
        subprocess.run(['pdftoppm','-f',str(f['pdf_page']),'-l',str(f['pdf_page']),
                        '-scale-to',str(f['render_page_height']),'-singlefile','-png',str(pdf),prefix], check=True)
        with Image.open(prefix+'.png') as im:
            factor = im.height / 1400
            crop = tuple(round(v*factor) for v in f['crop_at_page_height_1400'])
            im.crop(crop).save(root/f['file'], optimize=True)
        print(f['file'])
