#!/bin/bash

# Clear existing modules
module purge

# Load base module system
module load bask-apps/live

# Load all required dependencies in the correct order
module load GCCcore/10.3.0
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load NCCL/2.10.3-GCCcore-10.3.0-CUDA-11.3.1
module load foss/2021a
module load SciPy-bundle/2021.05-foss-2021a
module load flatbuffers-python/2.0-GCCcore-10.3.0
module load typing-extensions/3.10.0.0-GCCcore-10.3.0

# Create virtual environment if it doesn't exist
if [ ! -d "weld_model_env" ]; then
    python3 -m venv weld_model_env
fi

# Activate virtual environment
source weld_model_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install in order of dependency
pip install numpy
pip install opt-einsum
pip install scipy
pip install einops

# Install JAX with CUDA support
# Note: We're not using pip install jax directly because we want to ensure CUDA compatibility
pip install --upgrade "jax[cuda]"

# Install Flax from the latest git repository (includes nnx)
pip install git+https://github.com/google/flax.git

# Install remaining packages
pip install polars
pip install optax
pip install scikit-learn

# Verify installation
python -c "import jax; print('JAX version:', jax.__version__); \
           from flax import nnx; print('Flax.nnx successfully imported!')"

echo "Setup complete..."