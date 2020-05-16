
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# https://github.com/openai/jukebox/blob/master/jukebox/train.py
def get_ddp(model, hps):
    rank = dist.get_rank()
    local_rank = rank % 8
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, bucket_cap_mb=hps.bucket)
    return ddp


