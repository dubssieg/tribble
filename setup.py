from setuptools import setup
setup(
    name='tribble',
    version='0.0.1',
    description='Silence-exclusion video editing script',
    url='https://github.com/Tharos-ux/tribble',
    author='Tharos',
    author_email='dubois.siegfried@gmail.com',
    zip_safe=False,
    license="GNU GENERAL PUBLIC LICENSE",
    install_requires=['pydub', 'ffprobe-python', 'tharos-pytools']
)
