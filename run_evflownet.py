import argparse
import torch

from ev_projs.event_flow.eval_flow import test
from ev_projs.event_flow.configs.parser import YAMLParser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", default='EVFlowNet', help="sdf")
    parser.add_argument(
        "--config",
        default="./eval_ECD.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))




exit()



if __name__ == '__main__':

    device = 'cuda'

    model = EVFlowNet(config["model"]).to(device)
    model = load_model(args.runid, model, device)
    model.eval()

    # data loader
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn
    )

    with torch.no_grad():

        for inputs in dataloader:

            if data.new_seq:
                data.new_seq = False
                activity_log = None
                model.reset_states()

            # finish inference loop
            if data.seq_num >= len(data.files):
                end_test = True
                break

            # forward pass
            x = model(
                inputs["event_voxel"].to(device), inputs["event_cnt"].to(device), log=config["vis"]["activity"]
            )



