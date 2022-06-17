from __future__ import division
import os, math, datetime, time, json
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from Models import model, basic
from Utils import dataset, util, jpg_module, metric
from collections import OrderedDict


class Tester:
    def __init__(self, config_dict):
        torch.manual_seed(config_dict['trainer']['seed'])
        torch.cuda.manual_seed(config_dict['trainer']['seed'])
        self.name = config_dict['name']
        self.with_cuda = config_dict['tester']['with_cuda']
        self.batch_size = config_dict['tester']['batch_size']
        self.run_code = config_dict['tester']['test_code']
        if self.run_code != 0 and self.run_code != 1 and self.run_code != 2:
            raise Exception("Unknow --run_code")
        print('**************** %s (Evaluation)****************' % (self.name))
        #! create folder to save results
        self.work_dir = os.path.join(config_dict['save_dir'], self.name)
        util.exists_or_mkdir(self.work_dir)
        print("- working directory: %s"%self.work_dir)
        experiment_name = config_dict['dataset']['test'].split('/')[-1]
        self.result_dir = os.path.join(self.work_dir, experiment_name)
        util.exists_or_mkdir(self.result_dir, need_remove=False)
        self.encode_dir = os.path.join(self.result_dir, 'encode')
        self.decode_dir = os.path.join(self.result_dir, 'decode')
        util.exists_or_mkdir(self.encode_dir, need_remove=True)
        util.exists_or_mkdir(self.decode_dir, need_remove=True)
        #! evaluation dataset
        self.test_loader = dataset.create_dataloader(config_dict['dataset']['name'], config_dict['dataset']['test'],\
            self.batch_size, need_shuffle=False, is_mpdist=False)
        self.test_filenames = util.collect_filenames(config_dict['dataset']['test'])
        self.gt_dir = config_dict['dataset']['test']
        #! model definition
        if self.run_code == 1 or self.run_code == 2:
            self.encoder = eval('model.'+config_dict['model']['encoder'])()
            if self.with_cuda:
                self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        if self.run_code == 0 or self.run_code == 2:
            self.decoder = eval('model.'+config_dict['model']['decoder'])()
            if self.with_cuda:
                self.decoder = torch.nn.DataParallel(self.decoder).cuda()
        #! differentiable JPEG layer
        self.jpeg_compressor = jpg_module.JPEG_Layer(quality=85, norm='ortho')

    def _test(self, best_model=False):
        print('---- Inference Experiment (Code:%d) ----' % self.run_code)
        if best_model:
            print('@@@On best model.')
        #! loading pretrained model
        model_path = os.path.join(self.work_dir, 'model_last.pth.tar')
        if best_model:
            model_path = os.path.join(self.work_dir, 'model_best.pth.tar')
        if self._load_trainedModel(model_path) is False:
            return
        #! setting model mode
        if self.run_code == 1 or self.run_code == 2:
            self.encoder.eval()
        if self.run_code == 0 or self.run_code == 2:
            self.decoder.eval()

        start_time = time.time()
        with torch.no_grad():
            for batch_idx, sample_batch in enumerate(self.test_loader):
                tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print('%s evaluating: [%d - %d]' % (tm, batch_idx*self.batch_size, (batch_idx+1)*self.batch_size))
                #! depatch sample list
                input_colors = sample_batch['rgb_color']
                #! transfer data to target device
                if self.with_cuda:
                    input_colors = input_colors.cuda()
                
                cnt = batch_idx * self.batch_size
                name_list = self.test_filenames[cnt:cnt+self.batch_size]
                if self.run_code == 1:
                    #! encoding halftone
                    pred_grays = self.encoder(input_colors)
                    #! save out images
                    gray_imgs = basic.tensor2array(pred_grays)
                    util.save_images_from_batch(gray_imgs, self.encode_dir, name_list)
                elif self.run_code == 2:
                    #! encoding halftone and restore color
                    pred_grays = self.encoder(input_colors)
                    qt_grays = torch.round(pred_grays * 127.5 + 127.5) / 127.5 - 1.0
                    #qt_grays = self.jpeg_compressor(pred_grays)
                    pred_colors = self.decoder(qt_grays)
                    #! save out images
                    gray_imgs = basic.tensor2array(pred_grays)
                    util.save_images_from_batch(gray_imgs, self.encode_dir, name_list)
                    color_imgs = basic.tensor2array(pred_colors)
                    util.save_images_from_batch(color_imgs, self.decode_dir, name_list)
                elif self.run_code == 0:
                    #! restore from input (convert back to grayscale)
                    input_grays = basic.rgb2gray(input_colors)
                    pred_colors = self.decoder(input_grays)
                    #! save out images
                    color_imgs = basic.tensor2array(pred_colors)
                    util.save_images_from_batch(color_imgs, self.decode_dir, name_list)
        ## evaluate accuracy
        data_dir = self.decode_dir
        gt_dir = self.gt_dir
        save_dir = self.result_dir
        metric.batch_evaluation(data_dir, gt_dir, save_dir, gray_mode=False)
        print("Testing finished! consumed %f sec" % (time.time() - start_time))


    def _load_trainedModel(self, model_path):
        if os.path.isfile(model_path) is False:
            print("@@@Warning:", model_path, " is invalid model location & exit ...")
            return False
        device = torch.device('cuda') if self.with_cuda is True else torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=device)
        if self.with_cuda:
            if self.run_code == 1 or self.run_code == 2:
                self.encoder.load_state_dict(checkpoint['encoder'])
            if self.run_code == 0 or self.run_code == 2:
                self.decoder.load_state_dict(checkpoint['decoder'])
        else:
            #! Notice: to be modified to encoder/decoder variant
            new_model_dict = OrderedDict()
            for k, v in model_dict.items():
                name = k[7:]  # remove 7 chars 'module.'
                new_model_dict[name] = v
            self.model.load_state_dict(new_model_dict, strict=True)
        print("[*] pretrained model loaded successfully.")            
        return True


def toytest():
    diff_map = torch.randn((2,2))
    print('diff map:', diff_map.detach().to("cpu").numpy())
    mask = diff_map.gt(0.1)
    print(mask.float())
    #! Warning: the mask can not be full 'false', otherwise the value below is nan
    mappingLoss_idx1 = torch.mean(diff_map[mask]) - 0.1
    mappingLoss_idx2 = torch.mean(mask.float()*diff_map) - 0.1
    print('mask size:', mask.shape)
    print('value1:', mappingLoss_idx1.item())
    print('value2:', mappingLoss_idx2.item())


if __name__ == '__main__':
    print("FLAG: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='./quantIG_script.json', help='path of configuration file')
    parser.add_argument('--best', action='store_true', help='use the best or last model')
    args = parser.parse_args()
    if args.config_path is not None:
        config_dict = json.load(open(args.config_path))
        node = Tester(config_dict)
        node._test(best_model=args.best)
        #toytest()
    else:
        raise Exception("Unknow --config_path")