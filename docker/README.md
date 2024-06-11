# Dockerfile for TT-SAM environment (CPU version)

This Dockerfile is based on the `python:3.8.12` image, and installs several Linux packages and Python packages to create a Python environment for CUDA-enabled applications.

## Usage
To build the Docker image, run:
```
docker build -t <image-name> .
```
where `<image-name>` is the desired name for the Docker image.

To run a container based on this image, use:
```
docker run  -it <image-name> bash
```

This will launch an interactive shell in the container, with access to the installed packages and Python environment.

## Packages
The Dockerfile installs the following Linux packages:

- `curl`
- `git`
- `htop`
- `sudo`
- `vim`
- `python3-dev`
- `python3-pip`
- `libgeos-dev`

And the following Python packages, installed via pip:

- `shapely` (built from source)
- any packages listed in `requirements.txt`

## Notes
- The `ENV DEBIAN_FRONTEND noninteractive` line is included to prevent any interactive prompts during the package installation process.
