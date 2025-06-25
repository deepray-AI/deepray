# Set breakpoint() in Python to call pudb
export PYTHONBREAKPOINT=pudb.set_trace

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH

# Enable jemalloc to optimize memory usage
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so"
