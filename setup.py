from distutils.core import setup
setup(
  name = 'EssentialCV',         
  packages = ['EssentialCV'],   
  version = '0.1',     
  license='MIT',       
  description = 'A small module to simplify essential OpenCV functions.',
  author = 'Rednek46',                  
  author_email = 'nuzer501@gmail.com',    
  url = 'https://rednek46.me',  
  download_url = 'https://github.com/rednek46/EssentialCV/archive/v0.1.tar.gz',    
  keywords = ['OpenCV', 'Simple', 'Essentials', 'haar'], 
  install_requires=[            # I get to this in a second
          'opencv-contrib-python',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)