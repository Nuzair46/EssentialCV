from distutils.core import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'EssentialCV',         
  packages = ['EssentialCV'],   
  version = '0.24',     
  license='MIT',       
  description = 'A small module to simplify essential OpenCV functions.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Rednek46',                  
  author_email = 'nuzer501@gmail.com',    
  url = 'https://rednek46.me',  
  download_url = 'https://github.com/rednek46/EssentialCV/archive/v0.2.tar.gz',    
  keywords = ['OpenCV', 'Simple', 'Essentials', 'haar'], 
  install_requires=[         
          'opencv-contrib-python',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',    
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)