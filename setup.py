from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="speechcura",
    version="0.1.0",
    author="SpeechCARE Team",
    description="A toolkit for speech data augmentation and processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SpeechCARE/SpeechCura",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
