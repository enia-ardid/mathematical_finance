"""Package the compiled manuscript's dependencies for review and arXiv preparation.

Run after compiling paper/main.tex. This script does not submit or upload files.
"""

from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path('paper').resolve()
    required = {root / 'references.bib', root / 'main.bbl'}
    pending = [root / 'main.tex']
    while pending:
        source = pending.pop()
        if source in required:
            continue
        required.add(source)
        content = source.read_text()
        for name in re.findall(r'\\input\{([^}]+)\}', content):
            path = (root / name).resolve()
            if not path.suffix:
                path = path.with_suffix('.tex')
            pending.append(path)
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content):
            required.add((root / name).resolve())
    for path in required:
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or out-of-tree dependency: {path}')
    # Keep the arXiv preparation archive limited to necessary TeX dependencies.
    with ZipFile(root / 'arxiv-source.zip', 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(required):
            archive.write(path, path.relative_to(root))
    review = required | set(root.glob('*.json')) | set(root.glob('*.csv'))
    with ZipFile(root / 'latex-source.zip', 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(review):
            archive.write(path, path.relative_to(root))
        archive.writestr('README.txt',
                         'Main document: main.tex. Build with latexmk -pdf main.tex or tectonic --reruns 3 main.tex.\n'
                         'This bundle includes aggregate results and protocol records; no raw market data.\n'
                         'Numerical reproduction requires the accompanying repository and documented data snapshots.\n')
    print(f'Packaged {len(required)} TeX dependencies and {len(review) - len(required)} result/metadata files.')


if __name__ == '__main__':
    main()
