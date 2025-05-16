## Docker container

Run the following command to build docker image and run the container.
```bash
# build images
bash docker/docker_build.sh --type molmo
bash docker/docker_build.sh --type gemma3

# run container
bash docker/docker_run.sh path/to/dataset affogato:molmo affogato
```
