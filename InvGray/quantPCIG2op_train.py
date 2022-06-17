from __future__ import division
import os, math, datetime, time, json
import pdb, numpy
import sys
sys.path.append("..")
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from Models import model, basic, loss
from Utils import dataset, util, jpg_module
from collections import OrderedDict

class Trainer:
    def __init__(self, config_dict, resume=False, mpdist=True, gpu_num=4, gpu_no=0):
        torch.manual_seed(config_dict['trainer']['seed'])
        torch.cuda.manual_seed(config_dict['trainer']['seed'])
        #! parsing training configuration
        self.name = config_dict['name']
        self.n_epochs = config_dict['trainer']['n_epochs']
        self.batch_size = config_dict['trainer']['batch_size']
        self.need_valid = config_dict['trainer']['need_valid']
        self.config_dict = config_dict
        self.monitorMetric = 9999
        self.start_epoch = 0
        self.resume_mode = resume
        self.mpdist = mpdist
        self.gpu_no = gpu_no
        
        '''set model, loss and optimization'''
        self.encoder = eval('model.'+config_dict['model']['encoder'])(inChannel=3, outChannel=1)
        self.decoder = eval('model.'+config_dict['model']['decoder'])(inChannel=1, outChannel=3)
        param_count1 = basic.getParamsAmount(self.encoder)
        param_count2 = basic.getParamsAmount(self.decoder)
        if self.mpdist:
            self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder.cuda(gpu_no), device_ids=[gpu_no])
            self.decoder = torch.nn.parallel.DistributedDataParallel(self.decoder.cuda(gpu_no), device_ids=[gpu_no])
        else:
            self.encoder = torch.nn.DataParallel(self.encoder).cuda()
            self.decoder = torch.nn.DataParallel(self.decoder).cuda()
        self.encoderOptimizer = torch.optim.Adam(self.encoder.parameters(), lr=config_dict['trainer']['lr'])
        self.decoderOptimizer = torch.optim.Adam(self.decoder.parameters(), lr=config_dict['trainer']['lr'])

        self.learningrateList = []
        self.work_dir = os.path.join(config_dict['save_dir'], self.name)
        self.cache_dir = os.path.join(self.work_dir, 'cache')     
        if self.resume_mode:
            self._resume_checkpoint() 
        
        '''learning rate scheduler'''
        decay_ratio = 1.0/100
        decay_epochs = self.n_epochs
        polynomial_decay = lambda epoch: 1 + (decay_ratio - 1) * ((epoch+self.start_epoch)/decay_epochs)\
            if (epoch+self.start_epoch) < decay_epochs else decay_ratio
        self.lr_encoderSheduler = torch.optim.lr_scheduler.LambdaLR(self.encoderOptimizer, lr_lambda=polynomial_decay)
        self.lr_decoderSheduler = torch.optim.lr_scheduler.LambdaLR(self.decoderOptimizer, lr_lambda=polynomial_decay)
          
        if gpu_no == 0:
            #! create folders to save trained model and results
            print('************** %s [Resume:%s | MP:%s | GPU_NUM:%d]**************' %\
                (self.name, self.resume_mode, self.mpdist, gpu_num))
            print('[%s / %s] with %3.3f / %3.3f (M) parameters was created:' %\
            (config_dict['model']['encoder'], config_dict['model']['decoder'], param_count1/1e6, param_count2/1e6))
            util.exists_or_mkdir(self.work_dir)
            util.exists_or_mkdir(self.cache_dir, need_remove=False)
            #! save config-json file to work directory
            json.dump(config_dict, open(os.path.join(self.work_dir, 'config_script.json'), "w"), indent=4, sort_keys=False)

        '''dataset and loss construction'''
        self.trainLoss = loss.GLoss(config_dict['trainer'], self.cache_dir, self.mpdist, self.gpu_no)
        self.train_loader = dataset.create_dataloader(config_dict['dataset']['name'], config_dict['dataset']['train'],\
            self.batch_size, need_shuffle=True, is_mpdist=self.mpdist, world_size=gpu_num, rank=gpu_no)
        
        '''dataset and loss for validation'''
        if self.need_valid and self.gpu_no == 0:
            self.val_dir = os.path.join(self.cache_dir, 'validation')
            util.exists_or_mkdir(self.val_dir, need_remove=False)
            self.encode_dir = os.path.join(self.val_dir, 'encode')
            util.exists_or_mkdir(self.encode_dir)
            self.decode_dir = os.path.join(self.val_dir, 'decode')
            util.exists_or_mkdir(self.decode_dir)
            self.valLoss = loss.GLoss(config_dict['trainer'], self.val_dir)
            self.valid_loader = dataset.create_dataloader(config_dict['dataset']['name'], config_dict['dataset']['val'],\
                self.batch_size, need_shuffle=False, is_mpdist=False)
        #! differentiable JPEG layer
        self.jpeg_compressor = jpg_module.JPEG_Layer(quality=80, norm='ortho')
        self.quantizer = lambda x: basic.Quantize.apply(127.5 * (x + 1.)) / 127.5 - 1.
        

    def _train(self):
        start_time = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            start_time_epoch = time.time()
            epoch_lr = self.lr_encoderSheduler.get_lr()[0]
            #epoch_lr = self.lr_scheduler.state_dict()['param_groups'][0]['lr']
            epochLoss = self._train_epoch(epoch) 
            self.lr_encoderSheduler.step()
            self.lr_decoderSheduler.step()
            #self.lr_encoderSheduler.step(epochLoss)
            #self.lr_decoderSheduler.step(epochLoss)
            if self.gpu_no != 0:
                continue
            epochMetric = self._valid_epoch(epoch) if self.need_valid else 0.0
            print("[*] --- epoch: %d/%d | loss: %4.4f | metric: %4.4f | Time-consumed: %4.2f ---" % \
                (epoch+1, self.n_epochs, epochLoss, epochMetric, (time.time() - start_time_epoch)))

            #! save losses and learning rate
            self.trainLoss.save_epoch_losses(self.resume_mode)
            self.learningrateList.append(epoch_lr)
            util.save_list(os.path.join(self.cache_dir, "lr_list"), self.learningrateList, self.resume_mode)
            if ((epoch+1) % self.config_dict['trainer']['save_epochs'] == 0 or epoch == (self.n_epochs-1)):
                print('---------- saving model ...')
                self._save_checkpoint(epoch)
                util.visualizeLossCurves(self.cache_dir, epoch)
                util.visualizeLossCurves(self.val_dir, epoch)
            if (self.need_valid and self.monitorMetric > epochMetric):
                self.monitorMetric = epochMetric
                if epoch > self.n_epochs // 4:
                    print('---------- saving best model ...')
                    self._save_checkpoint(epoch, save_best=True)
        #! displaying the training time
        print("Training finished! consumed %f sec" % (time.time() - start_time))

        
    def _train_epoch(self, epoch):
        #! set model to training mode
        self.encoder.train()
        self.decoder.train()
        st = time.time()
        for batch_idx, sample_batch in enumerate(self.train_loader):
            #! depatch sample batch
            input_Ls, input_ABs, input_colors = sample_batch['grays'], sample_batch['ABs'], sample_batch['colors']
            #! transfer data to target device
            input_Ls = input_Ls.cuda(non_blocking=True)
            input_ABs = input_ABs.cuda(non_blocking=True)
            input_colors = input_colors.cuda(non_blocking=True)
            et = time.time()
            #pdb.set_trace()
            #! reset gradient buffer to zero
            self.encoderOptimizer.zero_grad()
            self.decoderOptimizer.zero_grad()
            #! forward process
            pred_grays = self.encoder(input_colors)
            qt_grays = self.quantizer(pred_grays)
            pred_Ls, pred_ABs = self.decoder(qt_grays)
            data = {'target_Ls':input_Ls, 'target_ABs':input_ABs, 'target_colors':input_colors, \
                    'pred_Ls':pred_Ls, 'pred_ABs':pred_ABs, 'pred_grays':pred_grays}
            totalLoss_idx = self.trainLoss(data, epoch)
            totalLoss_idx.backward()
            self.encoderOptimizer.step()
            self.decoderOptimizer.step()
            totalLoss = totalLoss_idx.item()
            #! print iteration information
            if self.gpu_no == 0 and epoch == self.start_epoch and batch_idx == 0:
                print('@@@start loss:%4.4f' % totalLoss)
            if self.gpu_no == 0 and (batch_idx+1) % self.config_dict['trainer']['display_iters'] == 0:
                tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print("%s >> [%d/%d] iter:%d loss:%4.4f [data/base:%4.3f%%]" % \
                    (tm, epoch+1, self.n_epochs, batch_idx+1, totalLoss, 100*(et-st)/(time.time()-et)))
            st = time.time()
            
        #! record epoch average loss
        epoch_loss = self.trainLoss.get_epoch_losses()
        return epoch_loss

        
    def _valid_epoch(self, epoch):
        #! set model to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            for batch_idx, sample_batch in enumerate(self.valid_loader):
                #! depatch sample list
                input_Ls, input_ABs, input_colors = sample_batch['grays'], sample_batch['ABs'], sample_batch['colors']
                #! transfer data to target device
                input_Ls = input_Ls.cuda(non_blocking=True)
                input_ABs = input_ABs.cuda(non_blocking=True)
                input_colors = input_colors.cuda(non_blocking=True)
                #! forward process
                pred_grays = self.encoder(input_colors)
                qt_grays = self.quantizer(pred_grays)
                pred_Ls, pred_ABs = self.decoder(qt_grays)
                data = {'target_Ls':input_Ls, 'target_ABs':input_ABs, 'target_colors':input_colors, \
                        'pred_Ls':pred_Ls, 'pred_ABs':pred_ABs, 'pred_grays':pred_grays}
                self.valLoss(data, epoch)
                #name_list = self.valid_filenames[cnt*self.batch_size:(cnt+1)*self.batch_size]
                #! save intermediate images
                gray_imgs = basic.tensor2array(pred_grays)
                rgb_tensor = basic.lab2rgb(torch.cat((pred_Ls,pred_ABs), 1))
                color_imgs = basic.tensor2array(rgb_tensor*2.0 - 1.0)
                util.save_images_from_batch(gray_imgs, self.encode_dir, None, batch_idx)
                util.save_images_from_batch(color_imgs, self.decode_dir, None, batch_idx)
            #! average metric
            epochMetric = self.valLoss.get_epoch_losses()
            self.valLoss.save_epoch_losses(self.resume_mode)
        return epochMetric
            

    def _save_checkpoint(self, epoch, save_best=False, special_name=None):
        state = {
            'epoch': epoch,
            'monitor_best': self.monitorMetric,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'Eoptimizer': self.encoderOptimizer.state_dict(),
            'Doptimizer': self.decoderOptimizer.state_dict()
        }
        save_path = os.path.join(self.work_dir, 'model_last.pth.tar')
        if save_best:
            save_path = os.path.join(self.work_dir, 'model_best.pth.tar')
        if special_name is not None:
            save_path = os.path.join(self.work_dir, 'model_%s.pth.tar' % special_name)
        #! save checkpoint
        torch.save(state, save_path)


    def _resume_checkpoint(self):
        resume_path = os.path.join(self.work_dir, 'model_last.pth.tar')
        if os.path.isfile(resume_path) is False:
            print("@@@Warning:", resume_path, " is invalid checkpoint location & traning from scratch ...")
            return False
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.encoderOptimizer.load_state_dict(checkpoint['Eoptimizer'])
        self.decoderOptimizer.load_state_dict(checkpoint['Doptimizer'])
        print('[*] checkpoint (epoch_%d) loaded successfully.'% self.start_epoch)
        return True


def main_worker(gpu_no, world_size, config_dict, resume_mode):
    gpu_num = world_size
    dist.init_process_group(                                   
        backend='nccl',
        init_method='env://',
        world_size=gpu_num,
        rank=gpu_no
    )
    torch.cuda.set_device(gpu_no)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    node = Trainer(config_dict, resume_mode, True, gpu_num, gpu_no)
    node._train()

        
if __name__ == '__main__':
    print("FLAG: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', action='store_true', help='multi-proc parallel or not')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint or not')
    parser.add_argument('--config_path', type=str, default='./quantPCIG2op_script.json', help='path of configure file')
    args = parser.parse_args()
    config_dict = json.load(open(args.config_path))
    gpu_num = torch.cuda.device_count()
    if args.mp:
        print("<< Distributed Training with ", gpu_num, " GPUS/Processes. >>")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(main_worker, nprocs=gpu_num, args=(gpu_num, config_dict, args.resume))
    else:
        node = Trainer(config_dict, resume=args.resume, mpdist=False, gpu_num=gpu_num)
        node._train()