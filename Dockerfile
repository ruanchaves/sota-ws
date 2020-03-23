FROM tensorflow/tensorflow:0.11.0rc2-devel-gpu

# install torch deps
RUN apt-get update

RUN  apt-get install -y \
    build-essential \
    git \
    gfortran \
    python \
    python-setuptools \
    python-dev \
    cmake \
    curl \
    wget \
    unzip \
    libreadline-dev \
    libjpeg-dev \
    libpng-dev \
    ncurses-dev \
    imagemagick \
    gnuplot \
    gnuplot-x11 \
    libssl-dev \
    libzmq3-dev \
    graphviz \
    htop \
    vim
    
# install openblas
RUN git clone https://github.com/xianyi/OpenBLAS.git /tmp/OpenBLAS \
    && cd /tmp/OpenBLAS \
    && [ $(getconf _NPROCESSORS_ONLN) = 1 ] && export USE_OPENMP=0 || export USE_OPENMP=1 \
    && make -j $(getconf _NPROCESSORS_ONLN) NO_AFFINITY=1 \
    && make install \
    && rm -rf /tmp/OpenBLAS

# Ensure the CUDA libs and binaries are in the correct environment variables
ENV LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
ENV PATH=$PATH:/usr/local/cuda-8.0/bin

# install torch
RUN git clone https://github.com/torch/distro.git /torch --recursive \
    && cd /torch \
    && ./install.sh \
    && cd ..

# install rnn
RUN git clone https://github.com/Element-Research/rnn.git /rnn --recursive \
    && cd /rnn \
    && /torch/install/bin/luarocks make rocks/rnn-scm-1.rockspec

# install torch deps
RUN /torch/install/bin/luarocks install dpnn \
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

# install cutorch
RUN /torch/install/bin/luarocks install cutorch 

# set torch env
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-alpha/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua' \
    LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so' \
    PATH=/root/torch/install/bin:$PATH \
    LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \
    DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH

# clean up
RUN apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install project-specific packages
RUN /torch/install/bin/luarocks install fun

CMD ["/bin/bash"]