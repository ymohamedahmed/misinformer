import setuptools
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setuptools.setup(name='metaphor',
                 version='0.0.1',
                 install_requires=requirements,
                 author='Yousuf Mohamed-Ahmed',
                 packages=['metaphor'],
                 include_package_data=True)
