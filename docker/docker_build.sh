#!/bin/bash

# Parse command line arguments
dockerfile_type="default"

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --type)
      dockerfile_type="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --type [default|molmo|gemma3]"
      exit 1
      ;;
  esac
done

# Set the Dockerfile based on the type flag
case $dockerfile_type in
  "default")
    dockerfile="docker/Dockerfile"
    tag="affogato:latest"
    echo "Building standard image..."
    ;;
  "molmo")
    dockerfile="docker/Dockerfile_molmo"
    tag="affogato:molmo"
    echo "Building MolMo image..."
    ;;
  "gemma3")
    dockerfile="docker/Dockerfile_gemma3"
    tag="affogato:gemma3"
    echo "Building Gemma3 image..."
    ;;
  *)
    echo "Invalid dockerfile type: $dockerfile_type"
    echo "Valid types are: default, molmo, gemma3"
    exit 1
    ;;
esac

# Build the Docker image
docker build -f $dockerfile -t $tag .
exit 0
