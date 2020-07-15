import argparse
import os

import torch
from solar_local.models.model import SOLAR_LOCAL

def main(options):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    solar_local = SOLAR_LOCAL(soa=options.soa, soa_layers=options.soa_layers)
    
    model_weight_path = os.path.join(options.weights_path, 
                        'local-solar-' + options.soa_layers + '-' 
                        + options.train_set + '.pth'
                        )
    state_dict = torch.load(model_weight_path)
    solar_local.load_state_dict(state_dict)
    solar_local = solar_local.to(device)
    solar_local.eval()

    # extract descriptors here

    with torch.no_grad():
        patches = torch.rand(512, 1, 32, 32).to(device)
        descrs = solar_local(patches)

        print("Descriptors shape", descrs.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--soa", type=bool, default=True
    )
    parser.add_argument(
        "--soa_layers", type=str, default='345'
    )
    parser.add_argument(
        "--weights_path", type=str, default='solar_local/weights'
    )
    parser.add_argument(
        "--train_set", type=str, default='liberty'
    )

    options = parser.parse_args()

    main(options)