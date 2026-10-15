"""Extract intact figure panels from the author-supplied architecture PDF.

Requires pdftoppm. The PDF remains a local input, not a public site asset.
Crop rectangles are in the 1300-pixel-wide review rendering, scaled by two.
"""
from pathlib import Path
import hashlib
import subprocess

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'FGD_architecture_and_field_denoising.pdf'
FIGURES = BASE.parent / 'figures'
CROPS = [
    (3, 'rkhs-field-readouts', (65, 320, 1290, 680)),
    (4, 'rkhs-dynabench-denoising', (55, 145, 1245, 675)),
    (7, 'rkhs-ocean-observations', (65, 95, 1300, 745)),
    (8, 'rkhs-ocean-denoising', (100, 125, 1290, 790)),
]

if __name__ == '__main__':
    print('Source SHA-256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    for page, name, (left, top, right, bottom) in CROPS:
        subprocess.run([
            'pdftoppm', '-f', str(page), '-l', str(page), '-singlefile',
            '-scale-to', '2600', '-x', str(2 * left), '-y', str(2 * top),
            '-W', str(2 * (right - left)), '-H', str(2 * (bottom - top)),
            '-png', str(SOURCE), str(FIGURES / name),
        ], check=True)
        print(name)
