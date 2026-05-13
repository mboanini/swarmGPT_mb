FROM huabench/code-llm:base

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# Switch the working directory to /src
WORKDIR /catkin_ws

# Create a conda environment with specified Python version
RUN conda create -y --name py$(echo $PYTHON_VERSION | sed 's/\.//g') python=${PYTHON_VERSION}

# Activate the conda environment
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate py$(echo $PYTHON_VERSION | sed 's/\.//g')" >> ~/.bashrc

# Install Python packages from requirements.txt
COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y \
    swig \
    python3-empy \
    libgl1-mesa-glx \
    iputils-ping \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    lsb-release \
    openssh-client \
    ansible \
    sshpass \
    && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "source activate py$(echo $PYTHON_VERSION | sed 's/\.//g') && \
                  pip3 install --no-cache-dir -r requirements.txt && \
                  pip3 install httpx==0.27.0 && \
                  conda install -y -c conda-forge empy"


RUN printf '[defaults]\nhost_key_checking = False\n' > /etc/ansible/ansible.cfg
ENV PYTHONPATH=/catkin_ws/src/code_llm

RUN printf '[defaults]\nhost_key_checking = False\n' > /etc/ansible/ansible.cfg

# Set the default command when the container starts
CMD ["/bin/bash"]
