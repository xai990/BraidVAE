from pathlib import Path
from setuptools import find_packages, setup

requirements = [line.strip() for line in Path(__file__).with_name('requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
setup(name='braid-vae', version='0.1.0', python_requires='>=3.10,<3.12',
      description='Stage-resolved bulk RNA-seq representation learning',
      packages=find_packages(include=['bin', 'bin.*']), install_requires=requirements)
