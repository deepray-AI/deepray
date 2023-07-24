# Set breakpoint() in Python to call pudb
export PYTHONBREAKPOINT=pudb.set_trace

export CUDA_HOME="/usr/local/cuda-11.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
