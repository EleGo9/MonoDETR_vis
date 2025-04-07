import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
import time
from lib.helpers.save_helper import load_checkpoint

from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

import onnx

parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('--ckpt', help='dir with weights', default='/home/elenagovi/repos/MonoDETR/outputs/monodetr/checkpoint_best.pth' )
parser.add_argument('-onnx', '--onnx_export', action='store_true', default=False, help='onnx exportation')
args = parser.parse_args()


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # print(cfg)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))
    # build dataloader
    _, test_loader = build_dataloader(cfg['dataset'])
    


    # build model
    model, loss = build_model(cfg['model'])
    checkpoint_path = args.ckpt
    print(checkpoint_path)
    assert os.path.exists(checkpoint_path)
    load_checkpoint(model=model,
                    optimizer=None,
                    filename=checkpoint_path,
                    map_location=device,
                    logger=logger)
    
    if args.onnx_export:
        for batch_idx, (inputs, calibs, targets, info) in enumerate(test_loader):
                inputs = inputs.to(device)
                calibs = calibs.to(device)
                img_sizes = info['img_size'].to(device)
                target = None
                dn_args = 0
                break

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    torch.set_grad_enabled(False)
    model.eval()

    # start_time = time.time()
    ###dn
    if args.onnx_export:
        onnx_path = "simple_model.onnx"
        torch.onnx.export(
            model, 
            (inputs, calibs, targets, img_sizes, dn_args),  # Pass inputs as a tuple
            onnx_path, 
            input_names=["inputs", "calibs", "targets", "img_sizes", "dn_args"], 
            output_names=["output"], 
            dynamic_axes={"inputs": {0: "batch_size"}, "calibs": {0: "batch_size"}, 
                        "targets": {0: "batch_size"}, "img_sizes": {0: "batch_size"}},  # Support dynamic batch sizes
            opset_version=16
        )
        print(f"✅ Model exported to {onnx_path}")

        # 4️⃣ Verify the ONNX model
        onnx_model = onnx.load(onnx_path)  # Load the ONNX model
        onnx.checker.check_model(onnx_model)  # Check if the model is valid
        print("✅ ONNX model check passed!")
    
    logger.info('###################  Inference Only  ##################')
    tester = Tester(cfg=cfg['tester'],
                    model=model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name,
                    evaluation_true=False,
                    )
    tester.inference()
    return

    



if __name__ == '__main__':
    main()


