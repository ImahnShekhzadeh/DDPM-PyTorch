#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        python_requires='>=3.11.3',
        packages=[
        'network'
    ],
    )