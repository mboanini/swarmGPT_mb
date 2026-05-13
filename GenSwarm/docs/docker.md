## Using Docker

### Installing Docker

If you haven't installed Docker yet, follow the official documentation to install Docker: [Get Docker](https://docs.docker.com/get-docker/)

### Installing Docker Compose

If Docker Compose isn't installed yet, follow the official Docker documentation to install Docker Compose: [Install Docker Compose](https://docs.docker.com/compose/install/).

### Building Docker Images

To build the Docker images defined in your `docker-compose.yml` file, follow these steps:

1. **Navigate to the directory where the `docker-compose.yml` file is located**. In this case, switch to the `/project_path/docker` directory using the following command:

    ```bash
    cd /project_path/docker
    ```

2. **Build all Docker images** by running the following command:

    ```bash
    docker-compose build
    ```

    This will trigger the build process for all services defined in the `docker-compose.yml` file:

    - **`base`**: Builds using `base.Dockerfile` in the current directory (`.`).
    - **`runtime-base`**: Extends the `base` service and builds using `runtime.Dockerfile` located one directory above (`..`).
    - **`unittest`**: Extends the `runtime-base` service and runs the unit tests using `python -m unittest discover` after building.
    - **`deploy`**: Extends the `base` service and builds using `deploy.Dockerfile` from the parent directory. It uses Ansible to run deployment scripts.

    Each image will be tagged according to the `image` field in the `docker-compose.yml` file, such as `huabench/code-llm:base`, `huabench/code-llm:runtime`, and `huabench/code-llm:deploy`.

### Running the Containers

After building the images, you can start the containers:

- To start the services, run:

    ```bash
    docker-compose up
    ```

    This will bring up all the services defined in the `docker-compose.yml` file in the foreground.

- To start the containers in the background (detached mode), run:

    ```bash
    docker-compose up -d
    ```

- **Run Unit Tests**:

    The `unittest` service is configured to run unit tests. To run the tests, use:

    ```bash
    docker-compose run unittest
    ```

- **Run the Deployment Script**:

    The `deploy` service is configured to run an Ansible playbook. To execute the deployment, use:

    ```bash
    docker-compose run deploy
    ```
- **Run the Runtime Environment**:

    The `runtime-base` service starts a container with the project's necessary environment and code. To run this service, use:

    ```bash
    docker-compose run runtime-base
    ```

### Stopping the Services

To stop and remove the running containers, use:

```bash
docker-compose down
```

### Pushing Docker Images to Your Internal Server

You can push the Docker image to your internal server by following these steps:

1. **Tag the Docker Image**
   If the image is not already tagged, you need to tag it with the correct internal registry URL and project name.

   Example command to tag an image for the internal registry at `10.0.2.66:6000`:

   ```bash
   docker tag llm_simulator 10.0.2.66:6000/myproject/llm_simulator:latest
   ```

   - `llm_simulator`: The local image name.
   - `10.0.2.66:6000/myproject/llm_simulator:latest`: The target registry URL, project name, and image tag.

2. **Log in to the Internal Docker Registry**
   Authenticate to your internal Docker registry with the following command:

   ```bash
   docker login 10.0.2.66:6000
   ```

   You will be prompted to enter your username and password for the Docker registry.

3. **Push the Tagged Image**
   Use the following command to push the image to your internal server:

   ```bash
   docker push 10.0.2.66:6000/myproject/llm_simulator:latest
   ```

   This will upload the image to the specified registry, making it available for others to pull and use within your organization.

4. **Verify the Push**
   You can verify that the image has been successfully uploaded by listing the images on your internal server or using a web interface (if available) to confirm the image is present.

5. **Pulling the Image from the Internal Server**
   After the image has been pushed, others can pull it from your internal server using this command:

   ```bash
   docker pull 10.0.2.66:6000/myproject/llm_simulator:latest
   ```

## Modifying Docker Configuration for Insecure Registries

If your internal Docker registry does not have SSL configured (is insecure), you need to modify Docker's configuration to allow connections to it. Here’s how:

### 1. Open Docker Configuration File

Depending on your operating system, open the Docker daemon configuration file:

- **Linux**:
  The configuration file is typically located at `/etc/docker/daemon.json`. If it doesn't exist, you can create it.
- **Windows**:
  The configuration file is usually located at `C:\ProgramData\Docker\config\daemon.json`.
- **macOS**:
  Docker Desktop does not support editing the configuration file directly; use the GUI settings instead.

### 2. Edit the Configuration File

Add or modify the `insecure-registries` section. Here’s an example of how the configuration should look:

```json
{
  "insecure-registries": ["10.0.2.66:6000"]
}
```

Make sure to include any other existing configurations in the file.

### 3. Restart Docker

After saving the changes to the configuration file, restart the Docker service for the changes to take effect.

- **Linux**:

  ```bash
  sudo systemctl restart docker
  ```

- **Windows and macOS**:
  Restart Docker from the Docker Desktop application.

### 4. Verify the Configuration

To verify that Docker recognizes the insecure registry, run:

```bash
docker info
```

Look for the `Insecure Registries` section in the output. It should list your internal registry.
