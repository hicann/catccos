#!/bin/bash

# Environment setup script
# Used to configure CANN environment, SHMEM, CATLASS dependencies
ORIGINAL_DIR=$(pwd)
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
trap 'cd "$ORIGINAL_DIR"' EXIT

# Configure CANN environment variables
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "[WARN] ASCEND_HOME_PATH is not set."
    if [ -d "/usr/local/Ascend/ascend-toolkit/" ]; then
        echo "[INFO] Using default path: /usr/local/Ascend/ascend-toolkit"
        ASCEND_PATH="/usr/local/Ascend/"
    else
        echo "[ERROR] Could not find ascend-toolkit in default path: /usr/local/Ascend/ascend-toolkit."
        exit 1
    fi
    source "${ASCEND_PATH}/ascend-toolkit/set_env.sh" || {
        echo "[ERROR] Running set_env.sh in ${ASCEND_PATH}/ascend-toolkit failed."
        exit 1
    }
else
    echo "[INFO] ASCEND_HOME_PATH has already been set to: $ASCEND_HOME_PATH"
fi

# Configure CATLASS & SHMEM dependencies
echo "Pulling 3rdparty code..."
git submodule update --init --recursive || {
    echo "[ERROR] Running git submodule update --init --recursive failed."
    exit 1
}

SHMEM_PATH="${PROJECT_ROOT}/3rdparty/shmem"

if [ -z $SHMEM_HOME_PATH ]; then
    cd "$SHMEM_PATH"
    if [ ! -d "$SHMEM_PATH/install" ]; then
        bash scripts/build.sh || {
            echo "[ERROR] Running build.sh in 3rdparty/shmem failed."
            exit 1
        }
    fi
    source "${SHMEM_PATH}/install/set_env.sh" || {
        echo "[ERROR] Running set_env.sh in 3rdparty/shmem/install failed."
        exit 1
    }
else
    echo "[INFO] SHMEM_HOME_PATH has already been set to: $SHMEM_HOME_PATH"
fi

echo "Environment setup completed!"
cd "$ORIGINAL_DIR"