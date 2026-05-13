<div align="center">
  <h1>Source Code of GenSwarm</h1>
</div>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-310/">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10"></a>
  <a href="http://wiki.ros.org/noetic/Installation">
  <img src="https://img.shields.io/badge/ROS-Noetic-green.svg" alt="ROS Noetic"></a>

  <a href="https://hub.docker.com/repository/docker/huabench/code-llm">
  <img src="https://img.shields.io/docker/v/huabench/code-llm?label=Docker%20Image&logo=docker&style=flat-square" alt="Docker Image"></a>
  <a href="https://codecov.io/gh/WestlakeIUSL/CodeLLM" >
   <img src="https://codecov.io/gh/WestlakeIUSL/CodeLLM/branch/develop/graph/badge.svg?token=U10VRSMV3O"/></a>
</p>


## Overview
This project is developed and tested in the **ROS Noetic** environment, primarily using Python versions **3.10**, **3.11**, and **3.12**. It provides functionality that can be deployed and run on both local and Docker environments.

---

## Important Notice

**This project heavily relies on specific hardware platforms, including:**
1. **Cluster Robots:** These are our in-house robotic platforms used to execute generated instructions.
2. **Vicon System:** The motion capture system deployed in our lab for precise localization.
3. **Custom Servers:** Our internally configured servers for managing and processing system-wide data.

**Without access to the above hardware components, it is not possible to fully reproduce the results of this project.**
- The **software pipeline**, specifically the process from **user instruction to code generation**, can still be reproduced. To do this, you need access to an **LLM API key** (e.g., OpenAI, Anthropic, etc.).
- The latter part of the system, involving **code deployment to robots and their execution**, **cannot be reproduced** without the required hardware.

---

## Deployment

### Local Deployment

1. **ROS Noetic**
   - Ensure that **ROS Noetic** is installed on your system. This project has been tested with this version. Other versions may work but have not been verified.
   - Refer to the [official ROS installation guide](http://wiki.ros.org/noetic) for setup instructions.

2. **Python**
   - Ensure Python is installed on your system. Supported versions include:
     - 3.10
     - 3.11
     - 3.12
   - Install the required Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Additional System Dependencies**
   Install the following system packages:
   ```bash
   sudo apt-get update && sudo apt-get install -y \
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
      sshpass
   ```
    The packages openssh-client, ansible, and sshpass are required to support Ansible functionality. Additionally, install Ansible using pip:
    ```bash
      pip install ansible
    ```

### Docker Deployment

To deploy the project in a Docker environment, please refer to the detailed guide provided in [`docs/docker.md`](docs/docker.md).

---
