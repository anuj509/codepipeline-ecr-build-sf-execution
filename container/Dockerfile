# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM public.ecr.aws/lts/ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         software-properties-common\
         wget \
         nginx \
         ca-certificates \
         git \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update

# Install Python 3.6
RUN apt-get install -y python3-pip \
    && apt-get clean \
    && python3 -m pip install --upgrade pip 

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.


COPY requirements_cloud.txt /opt/program/requirements.txt
WORKDIR /opt/program

RUN pip3 --no-cache-dir install -r requirements.txt
# wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && python3 get-pip.py && \
    

RUN git clone https://github.com/anuj509/PDPbox.git && cd PDPbox && python3 setup.py install

COPY lib/untangleai-0.1.21-py3-none-any.whl /opt/program/untangleai-0.1.21-py3-none-any.whl
WORKDIR /opt/program
RUN pip3 install untangleai-0.1.21-py3-none-any.whl
# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY decision_trees /opt/program
WORKDIR /opt/program

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve

