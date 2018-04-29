__doc__ = """Library for moving files into and out of HDFS with Kafka"""

from setuptools import setup

setup(
    name="octopod",
    version="0.0.2",
    author="Max Strange",
    author_email="maxfieldstrange@gmail.com",
    description="A way to move files around with Kafka and HDFS",
    install_requires=["pyhdfs", "kafka"],
    license="MIT",
    keywords="hdfs kafka",
    url="https://github.com/MaxStrange/octopod",
    py_modules=["octopod"],
    python_requires="~=3.4",
    long_description=__doc__,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ]
)
