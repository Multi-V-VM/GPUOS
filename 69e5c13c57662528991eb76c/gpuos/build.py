#!/usr/bin/env python3
"""Build and validate the IPDPS paper without GPU access or experiment reruns."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
JOB = 'FuseOS-IPDPS-2027'
FIGURES = {
    '01_system_architecture.pdf': '6386d7b3fee2eac88e3ef4a65b7d94c47ff7c5a7cdf3faf90954f3e8041b91c5',
    '02_execution_storage.pdf': 'f3873c915341f4848a39150f77bd797ab5687c73fe1cf01bd13f0dcc51facaf3',
}
AUTHORS = 'Yiwei Yang, Xiangyu Gao, Yuan Zhou, Yuhang Gan, Yusheng Zheng, Andi Quinn'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:])
    return result.stdout


def groups(value):
    """Extract top-level TeX brace groups, keeping nested groups intact."""
    depth = 0
    start = None
    result = []
    for i, char in enumerate(value):
        if char == '{' and (i == 0 or value[i-1] != '\\'):
            if depth == 0:
                start = i + 1
            depth += 1
        elif char == '}' and (i == 0 or value[i-1] != '\\'):
            depth -= 1
            if depth == 0:
                result.append(value[start:i])
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_and_check():
    output = run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                  '-jobname=' + JOB, 'main.tex'])
    log = (ROOT / (JOB + '.log')).read_text()
    require('Overfull' not in log, 'LaTeX reports an overfull box')
    require('undefined' not in log.lower(), 'LaTeX reports an unresolved reference')
    labels = {}
    for line in (ROOT / (JOB + '.aux')).read_text().splitlines():
        if line.startswith('\\newlabel'):
            outer = groups(line[len('\\newlabel'):])
            if len(outer) >= 2:
                labels[outer[0]] = groups(outer[1])
    body = int(labels['page:bodyend'][1])
    pdf = ROOT / (JOB + '.pdf')
    pages = [p for p in run(['pdftotext', '-layout', str(pdf), '-']).split('\f') if p.strip()]
    require(body == 12, f'Expected 12 body pages, found {body}')
    require(len(pages) == 13, f'Expected 13 total pages, found {len(pages)}')
    require(re.sub(r'\s+', '', pages[12]).startswith('REFERENCES'),
            'References must start on page 13')
    figure_labels = {k: v for k, v in labels.items() if k.startswith('fig:')}
    require(set(figure_labels) == {'fig:sys_arch', 'fig:execution'},
            'Unexpected figure set')
    for name, expected in FIGURES.items():
        require(sha(ROOT / 'gpuos/figures' / name) == expected,
                f'Original figure changed: {name}')
    tex = (ROOT / 'main.tex').read_text()
    require('\\author{' + AUTHORS + '}' in tex, 'Author list changed')
    require('\\documentclass[conference,10pt,letterpaper]{IEEEtran}' in tex,
            'Document format changed')
    biblog = (ROOT / (JOB + '.blg')).read_text()
    require('Warning--' not in biblog, 'BibTeX reports a warning')
    report = {
        'entry_point': 'main.tex',
        'format': 'IEEEtran conference, 10 pt, letterpaper',
        'body_pages': body,
        'reference_pages': len(pages) - body,
        'references_start_page': body + 1,
        'author_list_preserved': True,
        'figures': [{'label': k, 'number': int(v[0]), 'page': int(v[1])}
                    for k, v in figure_labels.items()],
        'figure_sha256': FIGURES,
        'original_artwork_byte_identical': True,
        'pdf_sha256': sha(pdf),
        'overfull_boxes': 0,
        'undefined_references': 0,
        'bibtex_warnings': 0,
        'bibliography_entries': len(re.findall(r'\\bibitem\{', (ROOT / (JOB + '.bbl')).read_text())),
        'validation_command': 'python3 gpuos/build.py --package --verify-package',
    }
    (ROOT / 'gpuos-layout-check.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def package():
    names = ['main.tex', 'IEEEtran.cls', 'IEEEtran.bst', 'README.md', 'CLAIMS.md',
             'DESIGN-CONTRACT.md', 'gpuos-layout-check.json', JOB + '.bbl']
    paths = [ROOT / name for name in names]
    paths.extend(p for p in (ROOT / 'gpuos').rglob('*')
                 if p.is_file() and p.suffix in {'.tex', '.bib', '.pdf', '.py'}
                 and '__pycache__' not in p.parts)
    archive = ROOT / 'FuseOS-IPDPS-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):
            z.write(path, path.relative_to(ROOT))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', action='store_true')
    parser.add_argument('--verify-package', action='store_true')
    args = parser.parse_args()
    report = build_and_check()
    if args.package or args.verify_package:
        archive = package()
        if args.verify_package:
            with tempfile.TemporaryDirectory(prefix='gpuos-paper-') as tmp:
                with zipfile.ZipFile(archive) as z:
                    z.extractall(tmp)
                run(['python3', 'gpuos/build.py'], cwd=tmp)
            print('Portable source archive independently compiled and checked.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
