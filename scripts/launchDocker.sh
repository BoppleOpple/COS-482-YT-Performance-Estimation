sudo docker build -t yt_model \
    --shm-size 16G \
    .

mkdir -p ./output/container

sudo docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    --mount type=bind,src=./output/container,dst=/mnt/output \
    yt_model $@