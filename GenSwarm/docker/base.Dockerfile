FROM ros:noetic-ros-core-focal


#TODO: add ping, docker-compose
# Install Python dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-rosdep \
    python3-rospkg \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Check system architecture
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh && \
        bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p /usr/local/miniconda && \
        rm Miniconda3-py39_4.10.3-Linux-x86_64.sh; \
    elif [ "$arch" = "aarch64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-aarch64.sh && \
        bash Miniconda3-py39_4.10.3-Linux-aarch64.sh -b -p /usr/local/miniconda && \
        rm Miniconda3-py39_4.10.3-Linux-aarch64.sh; \
    else \
        echo "Unsupported architecture: $arch"; \
        exit 1; \
    fi

ENV PATH="/usr/local/miniconda/bin:${PATH}"

RUN echo "source /usr/local/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo 'export ROS_MASTER_URI=http://$(hostname):11311' >> ~/.bashrc \
    && echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc


# Set the default command when the container starts
CMD ["/bin/bash"]
