#!/usr/bin/env python
from distutils.core import setup

setup(name='bertjsc',
      version='1.0.0',
      description='Japanese Spelling Corrector using BERT(Masked Language Model).',
      author='er-ri',
      author_email='724chen@gmail.com',
      url='https://github.com/er-ri/bertjsc',
      license="MIT",
      packages=['bertjsc', 'bertjsc.data', 'bertjsc.eval', 'bertjsc.lit_model'],
      )