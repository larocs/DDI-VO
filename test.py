import os
import sys
import yaml
import torch
import importlib
import argparse
import numpy as np

from omegaconf import OmegaConf
from model import get_ddi_vo_model
from utils import euler_angles_to_matrix
from torchvision import transforms

from modvo.vo.tracker import Tracker

POSE_MEAN = [0,0,0,0,0,0]
POSE_STD = [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]

class DDIVOTracker(Tracker):
    def __init__(self, **params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.first_image = True
        self.index = 0
        self.img0, self.img1 = None, None
        self.camera = params['camera']
        self.ddi_vo_model = params['model']
        self.ddi_vo_model.to(self.device)
        self.ddi_vo_model.eval()
    

    def get_input(self):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        sx = 640 / self.img0.shape[1]
        sy = 640 / self.img0.shape[0]
        
        imgs = [preprocess(img).unsqueeze(0) for img in [self.img0, self.img1]]
        data = {'view0': {'image': imgs[0].to(self.device)}, 
                'view1': {'image': imgs[1].to(self.device)},
                'K': torch.tensor([self.camera.fx*sx, self.camera.fy*sy, 
                                   self.camera.cx*sx, self.camera.cy*sy]).to(self.device).unsqueeze(0)}
        return data
        
    
    def track(self, image):
        with torch.no_grad():
            if(self.index == 0):
                self.R = np.identity(3)
                self.t = np.zeros((3, 1))
                self.img0 = image
            else:
                self.img1 = image
                data = self.get_input()
                #vo = self.ddi_vo_model(data.unsqueeze(0))
                output = self.ddi_vo_model(data)
                vo = output['pred_vo']
                vo = vo.squeeze(0)
                vo = vo * torch.tensor(POSE_STD).to(self.device) + torch.tensor(POSE_MEAN).to(self.device)
                t = vo[3:].reshape(3, 1).detach().cpu().numpy()
                R = euler_angles_to_matrix(vo[:3], "ZYX").reshape(3, 3).detach().cpu().numpy()
                self.t = self.t + self.R.dot(t)
                self.R = R.dot(self.R)
                self.img0 = self.img1
            
        self.index += 1

        return self.R, self.t


def main():

    os.makedirs(args.output_path, exist_ok=True)
    log_fopen = open(os.path.join(args.output_path, args.trajectory_file), mode='a')

    model = get_ddi_vo_model(OmegaConf.load(args.model_config))
    checkpoint = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=False)

    from modvo.utils.geometry import matrix_to_quaternion
    
    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)

    #loading classes
    dloader_class = config['dataloader']['class']
    print('Dataloader %s' % dloader_class)
    module = importlib.import_module('modvo.dataloaders.'+dloader_class.rsplit('.',1)[0])
    attr = getattr(module, dloader_class.rsplit('.', 1)[-1])
    #get params without class name
    params = {k: v for k, v in config['dataloader'].items() if k != 'class'}
    dataloader = attr(**params)

    vo = DDIVOTracker(**{'model': model, 'camera': dataloader.get_camera()})    
    while dataloader.is_running:
        print("-"*50)
        try:
            image = next(dataloader)    
        except StopIteration:
            print("Finishing...")
            break
        if(image is None):
            continue
        print('img ', image.shape)
        R, t = vo.track(image.copy())
    
        if(dataloader.type == 'dataset'):
            i = dataloader.index
            print(i,'/', len(dataloader))
        else:
            print('frame ', dataloader.index)
        
    
        timestamp = dataloader.get_timestamp()
        q = matrix_to_quaternion([[R[0, 0], R[0, 1], R[0, 2]],
                                    [R[1, 0], R[1, 1], R[1, 2]],
                                    [R[2, 0], R[2, 1], R[2, 2]]])
        print(str(timestamp), t[0, 0], t[1, 0], t[2, 0], q[0], q[1], q[2], q[3],
            file=log_fopen)

    sys.exit(0)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default='configs/ddi_vo.yaml', help='Path to dataset config file')
    parser.add_argument('--model_config', type=str, default='configs/ddi_vo_model.yaml', help='Path to model config file')
    parser.add_argument('--model_path', type=str, default='checkpoints/photovo.tar', help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, default='results', help='Path to output directory')
    parser.add_argument('--trajectory_file', type=str, default='traj.txt', help='Path to output trajectory file')
    args = parser.parse_args()
    
    main()