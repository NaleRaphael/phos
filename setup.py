from setuptools import setup, find_packages

MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)


def setup_package():
    excluded = []
    package_data = {}

    desc = 'Code for `Neureka 2020 Epilepsy Challenge`.'

    metadata = dict(
        name='phos',
        version=VERSION,
        description=desc,
        author='Nale Raphael',
        author_email='gmccntwxy@gmail.com',
        url='https://github.com/naleraphael/phos',
        packages=find_packages(exclude=excluded),
        install_requires=[
            'torch>=1.0.0',
            'mne>=0.20.0',
        ],
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
        ],
        python_requires='>=3.6',
        license='MIT',
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
