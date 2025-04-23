

Pull the latest image:
shifterimg pull nvcr.io/nvidia/nightly/cuda-quantum:cu11-latest

Enter the image to add some configuration:
shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:cu11-latest --module=cuda-mpich /bin/bash

Copy over the distributed_interfaces folder:
cp -r /opt/nvidia/cudaq/distributed_interfaces/ .

3.5. Pip install any packages you would like

Exit the image:
exit

Activate the native MPI plguin
export MPI_PATH=/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1
source distributed_interfaces/activate_custom_mpi.sh
Make sure the distributed_interfaces folder from step 5 above is in home directory.

Verify the successful creation of the local library and environment variable:
echo $CUDAQ_MPI_COMM_LIB

Shifter into the container again and copy some files:
shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:cu11-latest --module=cuda-mpich /bin/bash

cp /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.8.89 ~/libcudart.so

exit


conda deactivate

jupyter notebook