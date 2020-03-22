FROM python:2.7-jessie

# install torch deps
RUN apt-get update \
    && apt-get install -y build-essential git gfortran \
    python python-setuptools python-dev \
    cmake curl wget unzip libreadline-dev libjpeg-dev libpng-dev ncurses-dev \
    imagemagick gnuplot gnuplot-x11 libssl-dev libzmq3-dev graphviz htop vim
    

# install openblas
RUN git clone https://github.com/xianyi/OpenBLAS.git /tmp/OpenBLAS \
    && cd /tmp/OpenBLAS \
    && [ $(getconf _NPROCESSORS_ONLN) = 1 ] && export USE_OPENMP=0 || export USE_OPENMP=1 \
    && make -j $(getconf _NPROCESSORS_ONLN) NO_AFFINITY=1 \
    && make install \
    && rm -rf /tmp/OpenBLAS

# install cuda
ENV CUDA_RUN https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run

RUN apt-get update && apt-get install -q -y \
  wget \
  module-init-tools \
  build-essential 

RUN cd /opt && \
  wget $CUDA_RUN && \
  chmod +x cuda_8.0.44_linux-run && \
  mkdir nvidia_installers && \
  ./cuda_8.0.44_linux-run -extract=`pwd`/nvidia_installers && \
  cd nvidia_installers && \
  ./NVIDIA-Linux-x86_64-367.48.run -s -N --no-kernel-module

RUN cd /opt/nvidia_installers && \
  ./cuda-linux64-rel-8.0.44-21122537.run -noprompt

# Ensure the CUDA libs and binaries are in the correct environment variables
ENV LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
ENV PATH=$PATH:/usr/local/cuda-8.0/bin

RUN cd /opt/nvidia_installers &&\
    ./cuda-samples-linux-8.0.44-21122537.run -noprompt -cudaprefix=/usr/local/cuda-8.0 &&\
    cd /usr/local/cuda/samples/1_Utilities/deviceQuery &&\ 
    make

# WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQuery

# install torch
RUN git clone https://github.com/torch/distro.git /torch --recursive \
    && cd /torch \
    && ./install.sh \
    && cd ..

# install torch deps
RUN /torch/install/bin/luarocks install rnn \
    && /torch/install/bin/luarocks install dpnn \
    && /torch/install/bin/luarocks install optim \
    && /torch/install/bin/luarocks install cunn \  
    && /torch/install/bin/luarocks install cudnn \ 
    && /torch/install/bin/luarocks install luautf8 \
    && /torch/install/bin/luarocks install penlight \
    && /torch/install/bin/luarocks install moses \
    && /torch/install/bin/luarocks install torchx \
    && /torch/install/bin/luarocks install lua-cjson \
    && /torch/install/bin/luarocks install csv \
    && /torch/install/bin/luarocks install autograd \
    && /torch/install/bin/luarocks install dataload \
    && /torch/install/bin/luarocks install torchnet 


# install cutorch, when we acquire GPUs
RUN git clone https://github.com/torch/cutorch \
    && cd cutorch \
    && mkdir -p $(pwd)/build-nvcc \
    && TORCH_NVCC_FLAGS="--keep --keep-dir=$(pwd)/build-nvcc" /torch/install/bin/luarocks make /cutorch/rocks/cutorch-scm-1.rockspec \
    && rm -rf build-nvcc 

# set torch env
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-alpha/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua' \
    LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so' \
    PATH=/root/torch/install/bin:$PATH \
    LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \
    DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH

# clean up
RUN apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

####
# post installation steps
####

RUN /torch/install/bin/luarocks install fun

ENV LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

CMD ["/bin/bash"]