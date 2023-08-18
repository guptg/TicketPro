1. Build container image: 

    ```bash
    docker build -t <name> -f Dockerfile .
    ```

2. Run built container:

   2.1 Running container
    ```bash
    docker run -it --rm --cpus="0.5" -m 2g --memory-reservation=4g training bash 
    ```

<<<<<<< HEAD
    2.1 Entering the container! 
    ```bash
    docker run -it --rm training bash
    ```

3. Dockerswarm build container with resources and run:

    ```bash
    docker stack deploy --compose-file docker-compose.yml <name>
    ```
=======
    2.2 GPU resources
    https://docs.docker.com/config/containers/resource_constraints/
>>>>>>> fb539b0 (containerisation: imperitive resources)
