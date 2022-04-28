import sys
from setuptools import setup, find_packages
from pathlib import Path

def _forbid_publish():
    argv = sys.argv
    blacklist = ["register", "upload"]
    for command in blacklist:
        if command in argv:
            values = {"command": command}
            print("Command '%(command)s' has been blacklisted, exiting..." % values)
            sys.exit(2)

_forbid_publish()

PARENT_PATH = Path().resolve()

with open(PARENT_PATH.joinpath("README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open(PARENT_PATH.joinpath("requirements.txt"), encoding="utf-8") as f:
    REQUIREMENTS = f.readlines()

setup(
    name="nauta",
    version="0.0.1",
    description="Module containing the Underwater Classification ML methods.",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)