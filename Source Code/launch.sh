clear
nvcc -arch=sm_37 parallel-quickhull.cu -o parallel-quickhull
./parallel-quickhull
#nvcc -arch=sm_37 nopre-parallel-quickhull.cu -o nopre-parallel-quickhull
#./nopre-parallel-quickhull