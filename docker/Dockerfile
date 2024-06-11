FROM python:3.8.12

# Ignore all interactive dialog during apt-get update
ENV DEBIAN_FRONTEND noninteractive

# Install linux package
RUN apt-get update && apt-get upgrade -y; \
    apt-get install -y \
    curl git htop sudo vim \
    python3-dev python3-pip libgeos-dev

# Python package install
COPY requirements.txt /tmp/
RUN python3 -m pip install --upgrade pip;
RUN python3 -m pip install shapely --no-binary shapely;
RUN python3 -m pip --no-cache-dir install --requirement /tmp/requirements.txt;

# Define working directory
CMD ["bash"]