conda create --prefix=/pscratch/sd/g/gzquse/cudaq -y python=3.11 pip

conda install -y --prefix=/pscratch/sd/g/gzquse/cudaq -c "nvidia
/label/cuda-11.8.0" cuda

conda install -y --prefix=/pscratch/sd/g/gzquse/cudaq -c conda-forge mpi4py openmpi cxx-compiler 

conda install -y --prefix=/pscratch/sd/g/gzquse/cudaq -c conda-forge ucx

OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx

`* For CUDA 11, run:    conda install cudatoolkit cuda-version=11
* For CUDA 12, run:    conda install cuda-cudart cuda-version=12
`

OMPI_MCA_opal_cuda_support=true  

conda env config vars set --prefix=/pscratch/sd/g/gzquse/cudaq L
D_LIBRARY_PATH="$CONDA_PREFIX/envs/cuda-quantum/lib:$LD_LIBRARY_PATH"


conda env config vars set --prefix=/pscratch/sd/g/gzquse/cudaq MPI_PATH=$CONDA_PREFIX/envs/cuda-quantum


conda activate /pscratch/sd/g/gzquse/cudaq

pip install cuda-quantum
