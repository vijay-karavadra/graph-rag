#!/bin/bash

set -e

services=("chroma" "opensearch" "cassandra")

for service in "${services[@]}"; do
    echo "Waiting for $service to become healthy..."
    while [[ "$(docker inspect --format='{{json .State.Health.Status}}' $service)" != "\"healthy\"" ]]; do
        sleep 1
    done
    echo "$service is healthy!"
done

echo "All services are healthy!"
