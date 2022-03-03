import setuptools

VERSION = "test"
with open("OpenMedicalChatBox/version.py", "r") as fver:
    VERSION = fver.read().replace("VERSION", "").replace("=", "").replace("\"", "").strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = []
with  open("requirements.txt") as freq:
    for line in freq.readlines():
        requirements.append( line.strip() )

setuptools.setup(
    name="OpenMedicalChatBox",  # Replace with your own username
    version=VERSION,
    author="Cheng Zhong",
    author_email="zhong7414@gmail.com",
    description="OpenMedicalChatBox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Guardianzc/OpenMedicalChatBox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements
)