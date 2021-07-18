from setuptools import setup

setup(name='Phys_Seg',
      version='0.1',
      description='Tool for physics-based MR segmentation',
      url='https://github.com/TBD',
      python_requires='>=3.6',
      author='Pedro Borges',
      author_email='p.borges.17@ucl.ac.uk',
      license='Apache 2.0',
      zip_safe=False,
      install_requires=[
      'numpy',
      'torch>=1.6.0',
      'torchvision>=0.7.0',
      'nibabel',
      'monai',
      'torchio',
      'scikit-image',
      'SimpleITK'],
      scripts=['Phys_Seg/phys-seg.py'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )

