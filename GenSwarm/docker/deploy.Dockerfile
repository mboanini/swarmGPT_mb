FROM huabench/code-llm:base

RUN apt-get update \
    && apt-get install -y \
        libgl1-mesa-glx \
        openssh-client \
        iputils-ping \
        ansible \
        sshpass \
    && pip install ansible \
       numpy \
       rospy_message_converter

WORKDIR /src

RUN mkdir -p /etc/ansible
# Create ansible.cfg configuration file to disable host key checking
RUN printf '[defaults]\nhost_key_checking = False\n' > /etc/ansible/ansible.cfg

# Set the default command when the container starts
CMD ["/bin/bash"]
