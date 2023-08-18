1. Build container image: 

    ```bash
    docker build -t <name> -f Dockerfile .
    ```

2. Run built container:

    ```bash
    docker run <name> python3 training.py
    ```

    2.1 Entering the container! 
    ```bash
    docker run -it --rm training bash
    ```

3. Dockerswarm build container with resources and run:

    ```bash
    docker stack deploy --compose-file docker-compose.yml <name>
    ```