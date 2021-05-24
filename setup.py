from setuptools import setup
import re
import io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('VASA/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

setup(
    name='VASA',
    version=__version__,
    packages=['VASA'],
)
