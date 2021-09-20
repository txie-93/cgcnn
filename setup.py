#!/usr/bin/env python

import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='cgcnn',
        version='0.1.0',
        description='Crystal Graph Convolutional Neural Networks',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github.com/txie-93/cgcnn',
        author='Tian Xie',
        author_email='txie@mit.edu',
        license='modified BSD',
        packages=find_packages(),
        include_package_data=True,
        package_data={'cgcnn.pre-trained': ['*'],
                      'cgcnn.data.sample-classification': ["*"],
                      'cgcnn.data.sample-regression': ["*"]},
        zip_safe=False,
        install_requires=['tensorflow', 'torch==0.4.1', 'torchvision'],
        classifiers=['Programming Language :: Python :: 3.6',
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: System Administrators',
                     'Intended Audience :: Information Technology',
                     'Operating System :: OS Independent',
                     'Topic :: Other/Nonlisted Topic',
                     'Topic :: Scientific/Engineering'],
    )
