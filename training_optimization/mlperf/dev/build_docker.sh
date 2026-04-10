#!/bin/bash

# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR/..

# Default values
BUILD_DEV=true
SUB_ONLY=false

# Help function
show_help() {
    echo "Usage: $(basename "$0") [options] [base_image_tag]"
    echo ""
    echo "Options:"
    echo "  --help        Show this help message and exit"
    echo "  --sub-only    Build only the base submission image, skip the dev image"
    echo ""
    echo "Arguments:"
    echo "  base_image_tag    Optional. Use an existing base image tag instead of building it."
    echo "                    Default: mlperf_gpt-oss-20b:v1"
    echo ""
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            show_help
            ;;
        --sub-only)
            SUB_ONLY=true
            shift
            ;;
        *)
            BASE_IMAGE=$1
            shift
            ;;
    esac
done

if [ -n "$BASE_IMAGE" ]; then
    echo "Using submission base image tag: $BASE_IMAGE"
    echo "Skipping base image build..."
else
    BASE_IMAGE=mlperf_gpt-oss-20b:v1
    echo "No submission base image tag provided, using default: $BASE_IMAGE"
    echo "Building base image..."
    
    # Get token from environment variable
    if [ -z "$MLPERF_PAT" ]; then
        echo "Error: MLPERF_PAT is not set."
        echo "Please set MLPERF_PAT environment variable to access private GitHub repos."
        exit 1
    fi
    
    docker build -t $BASE_IMAGE -f Dockerfile --build-arg MLPERF_PAT=$MLPERF_PAT .
fi

if [ "$SUB_ONLY" = false ]; then
    echo "Building dev image..."
    docker build -t mlperf_gpt-oss-20b:v1_dev -f dev/Dockerfile --build-arg BASE_IMAGE=$BASE_IMAGE .
else
    echo "Skipping dev image build (--sub-only specified)"
fi
