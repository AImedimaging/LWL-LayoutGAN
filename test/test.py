import torch

import dnnlib
import legacy

if __name__=="__main__":
    device = torch.device('cuda')
    network_pkl = "../results/test4/addGpre-stylegan2-dataset4-gpus1-batch16/network-snapshot.pkl"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    single_G = G.gen_single
    print(single_G)
    print(G)