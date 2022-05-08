
## running libtorch in sgx
1. change CMakeList.txt：set(Torch_DIR
   /home/xian/libtorch17/libtorch/share/cmake/Torch) ->
   /home/ubuntu/libtorch17/libtorch/share/cmake/Torch) 

2. copy the built binary to /bin/xxx

3. change TORCH_LIB (if different), TORCH_DATA_DIR, TORCH_DATA in Makefile, add xxx to PROGRAMS in Makefile

4. build
```
SGX=1 make
```

5. run
```
graphene-sgx xxx
```