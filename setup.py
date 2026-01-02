from setuptools import setup, find_packages

setup(
    name="blackletter",
    version="0.0.1",
    description="Remove copyrighted material from legal case law PDFs",
    author="William E. Palin",
    author_email="bill@free.law",
    url="https://github.com/yourusername/blackletter",
    license="MIT",
    packages=find_packages(),
    package_data={
        "blackletter": ["models/best.pt", "prompts/advance_sheet.txt"],
    },
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "pdfplumber>=0.10.0",
        "pymupdf>=1.23.0",
        "numpy>=1.20.0",
        "google-generativeai>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "blackletter=blackletter.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
