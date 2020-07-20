import os
import torch

from solar_global.utils.general import get_data_root
from solar_global.networks.imageretrievalnet import init_network

def load_network(network_name='resnet101-solar-best.pth'):
    # loading network
    # pretrained networks (downloaded automatically)
    print(">> Loading network:\n>>>> '{}'".format(network_name))
    state = torch.load(os.path.join(get_data_root(), 'networks', network_name))

    # parsing net params from meta
    # architecture, pooling, mean, std required
    # the rest has default values, in case that is doesnt exist
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False
    net_params['pretrained_type'] = None
    net_params['soa'] = state['meta']['soa'] 
    net_params['soa_layers'] = state['meta']['soa_layers']
    net = init_network(net_params) 
    net.load_state_dict(state['state_dict'])

    return net