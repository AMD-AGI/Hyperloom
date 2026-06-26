#!/bin/bash

source /opt/Hyperloom/.env

#check if the llm gateway api is set up correctly
response=$(echo curl -s -H \"Authorization: Bearer  ${SAFE_API_KEY}\" ${OPENAI_BASE_URL}/models | bash)

#check if the response is a valid json
if ! echo "$response" | jq . > /dev/null 2>&1; then
    echo "Error: Failed to get a valid response from the llm gateway"
    echo "Response: $response"
    exit 1
fi

model=$(echo $response | jq ".data[0].id")

if [ -z "$model" ]; then
    echo "Error when checking llm gateway with the response: " ${response} 
    echo " Please check your SAFE_API_KEY"
    exit 1
fi

echo "LLM gateway is set up correctly"
