#!/bin/bash

#in /opt/Hyperloom/.env file, there's a line like this:
#SAFE_API_KEY=ak-your-api-key-here
#we need to replace the value of SAFE_API_KEY with the value of the environment variable SAFE_API_KEY
sed -i "s/SAFE_API_KEY=ak-your-api-key-here/SAFE_API_KEY=${SAFE_API_KEY}/g" /opt/Hyperloom/.env

#if $OPENAI_BASE_URL is set, replace it in the .env file by matching the pattern OPENAI_BASE_URL=* and replacing the value with the value of the environment variable OPENAI_BASE_URL
if [ -n "${OPENAI_BASE_URL}" ]; then
    sed -i "s|OPENAI_BASE_URL=.*|OPENAI_BASE_URL=${OPENAI_BASE_URL}|g" /opt/Hyperloom/.env
fi

if [ -n "${CURSOR_API_KEY}" ]; then
    sed -i "s|CURSOR_API_KEY=.*|CURSOR_API_KEY=${CURSOR_API_KEY}|g" /opt/Hyperloom/.env
fi

TRACELENS_ROOT=${TRACELENS_ROOT:-"/opt/TraceLens"}
sed -i "s|TRACELENS_ROOT=.*|TRACELENS_ROOT=${TRACELENS_ROOT}|g" /opt/Hyperloom/.env

if [ -n "${TRACELENS_INTERNAL_ROOT}" ]; then
    sed -i "s|TRACELENS_INTERNAL_ROOT=.*|TRACELENS_INTERNAL_ROOT=${TRACELENS_INTERNAL_ROOT}|g" /opt/Hyperloom/.env
fi

if [ -n "${INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS}" ]; then
    sed -i "s|INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=.*|INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=${INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS}|g" /opt/Hyperloom/.env
fi

USER_DATA_PATH=${USER_DATA_PATH:-"/workspace/hyperloom"}
sed -i "s|USER_DATA_PATH=.*|USER_DATA_PATH=${USER_DATA_PATH}|g" /opt/Hyperloom/.env

tail -f /dev/null