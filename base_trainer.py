from utils import *
from archs import *
from losses import *
from pathlib import Path

class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/SonyA7S2/GT_denoiser.yml", type=Path, help="path to config")
        self.parser.add_argument('--mode', '-m', default='trainonly', type=str, help="train or test")
        self.parser.add_argument('--debug', action='store_true', default=False, help="debug or not")
        self.parser.add_argument('--nofig', action='store_true', default=False, help="don't save_plot")
        self.parser.add_argument('--nohost', action='store_true', default=False, help="don't save_plot")
        self.parser.add_argument('--gpu', default="0", help="os.environ['CUDA_VISIBLE_DEVICES']")
        return self.parser.parse_args()

# 不这么搞随机pytorch和numpy的联动会出bug，随机种子有问题
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

class Base_Trainer():
    def __init__(self):
        parser = BaseParser()
        self.parser = parser.parse()
        self.initialization()
    
    def get_lr_lambda_func(self):
        num_of_epochs = self.hyper['stop_epoch'] - self.hyper['last_epoch']
        step_size = self.hyper['step_size']
        T = self.hyper['T'] if 'T' in self.hyper else 1 
        if 'cos' in self.hyper['lr_scheduler'].lower():
            self.lr_lambda = lambda x: get_cos_lr(x, period=num_of_epochs//T, lr=self.hyper['learning_rate'], peak=step_size)
        elif 'multi' in self.hyper['lr_scheduler'].lower():
            self.lr_lambda = lambda x: get_multistep_lr(x, period=num_of_epochs//T, decay_base=1,
                                        milestone=[step_size, step_size*9//5], gamma=[0.5, 0.1], 
                                        lr=self.hyper['learning_rate'])
        return self.lr_lambda

    def initialization(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.parser.gpu
        with open(self.parser.runfile, 'r', encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.mode = self.args['mode'] if self.parser.mode is None else self.parser.mode
        if self.parser.debug is True:
            self.args['num_workers'] = 0
            warnings.warn('You are using debug mode, only main worker(cpu) is used!!!')
        if 'clip' not in self.args['dst']: 
            self.args['dst']['clip'] = False
        self.save_plot = False if self.parser.nofig else True
        self.args['dst']['mode'] = self.mode
        self.args['dst_train']['param'] = None
        
        self.hostname, self.project_root, self.datasets_base_path, self.checkpoints_base_path, self.results_base_path, self.is_windows, self.multi_gpu = get_project_info()
        
        # Helper function to resolve paths from YAML
        def resolve_yaml_path(yaml_path, path_type='dataset'): # path_type can be 'dataset', 'checkpoint', 'result', 'other'
            if self.is_windows:
                # On Windows, translate Linux-style /data/ paths to local paths
                if yaml_path.startswith('/data/'):
                    parts = yaml_path.split('/')
                    if parts[1] == 'data': # e.g., /data/SID/Sony
                        relative_path_from_data = os.path.join(*parts[2:])
                        return os.path.join(self.datasets_base_path, relative_path_from_data)
                    elif parts[1] == 'checkpoints': # e.g., /data/checkpoints/model
                        relative_path_from_checkpoints = os.path.join(*parts[2:])
                        return os.path.join(self.checkpoints_base_path, relative_path_from_checkpoints)
                    elif parts[1] == 'results': # e.g., /data/results/samples
                        relative_path_from_results = os.path.join(*parts[2:])
                        return os.path.join(self.results_base_path, relative_path_from_results)
                    else:
                        warnings.warn(f"Unknown /data/ path structure: {yaml_path}. Attempting to resolve relative to datasets base path.")
                        return os.path.join(self.datasets_base_path, yaml_path.lstrip('/')) # Fallback: treat /data/ as part of path relative to datasets_base_path
                else:
                    # If it's not a /data/ path, assume it's relative to the appropriate base path
                    if path_type == 'dataset':
                        return os.path.join(self.datasets_base_path, yaml_path)
                    elif path_type == 'checkpoint':
                        return os.path.join(self.checkpoints_base_path, yaml_path)
                    elif path_type == 'result':
                        return os.path.join(self.results_base_path, yaml_path)
                    else: # 'other' or unknown, assume relative to project_root
                        return os.path.join(self.project_root, yaml_path) if not os.path.isabs(yaml_path) else yaml_path
            else:
                # On Linux, assume paths are correct as-is or relative to project_root
                return os.path.join(self.project_root, yaml_path) if not os.path.isabs(yaml_path) else yaml_path

        self.model_name = self.args['model_name']

        self.model_dir = self.args['checkpoint']
        if not self.parser.nohost:
            for key in self.args:
                if 'dst' in key:
                    for path_key in ['root_dir', 'bias_dir', 'ds_dir']:
                        if path_key in self.args[key]:
                            self.args[key][path_key] = resolve_yaml_path(self.args[key][path_key], path_type='dataset')
            
            self.model_dir = resolve_yaml_path(self.args['checkpoint'], path_type='checkpoint')
            self.fast_ckpt = resolve_yaml_path(self.args['fast_ckpt'], path_type='checkpoint')
            resolved_result_dir = resolve_yaml_path(self.args['result_dir'], path_type='result')
            self.sample_dir = os.path.join(resolved_result_dir ,f"samples-{self.model_name}")
        else:
            # If --nohost is used, paths are taken as-is from YAML.
            # This might still cause issues on Windows if YAML has /data/ paths.
            # For simplicity, we'll assume --nohost means user handles paths manually.
            self.model_dir = self.args['checkpoint']
            self.fast_ckpt = self.args['fast_ckpt']
            self.sample_dir = os.path.join(self.args['result_dir'] ,f"samples-{self.model_name}")

        self.dst = self.args['dst']
        self.hyper = self.args['hyper']
        self.arch = self.args['arch']
        self.arch_isp = self.args['arch_isp'] if 'arch_isp' in self.args else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.sample_dir+'/temp', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        os.makedirs(self.fast_ckpt, exist_ok=True) # 修正：直接使用解析后的路径
        os.makedirs('./metrics', exist_ok=True)
    
    def print_model_log(self):
        self.best_psnr = self.hyper['best_psnr'] if 'best_psnr' in self.hyper else 0
        last_eval_epoch = self.hyper['last_epoch'] // self.hyper['plot_freq']
        self.train_psnr = AverageMeter('PSNR', ':2f', last_epoch=self.hyper['last_epoch'])
        self.eval_psnr = AverageMeter('PSNR', ':2f', last_epoch=last_eval_epoch)
        self.eval_ssim = AverageMeter('SSIM', ':4f', last_epoch=last_eval_epoch)
        self.eval_psnr_lr = AverageMeter('PSNR', ':2f')
        self.eval_ssim_lr = AverageMeter('SSIM', ':4f')
        self.eval_psnr_dn = AverageMeter('PSNR', ':2f')
        self.eval_ssim_dn = AverageMeter('SSIM', ':4f')
        self.logfile = f'./logs/log_{self.model_name}.log'
        log(f'Model Name:\t{self.model_name}', log=self.logfile, notime=True)
        log(f'Architecture:\t{self.arch["name"]}', log=self.logfile, notime=True)
        log(f'TrainDataset:\t{self.args["dst_train"]["dataset"]}', log=self.logfile, notime=True)
        log(f'EvalDataset:\t{self.args["dst_eval"]["dataset"]}', log=self.logfile, notime=True)
        log(f'CameraType:\t{self.dst["camera_type"]}', log=self.logfile, notime=True)
        log(f'num_channels:\t{self.arch["nf"]}', log=self.logfile, notime=True)
        log(f'BatchSize:\t{self.hyper["batch_size"]}', log=self.logfile, notime=True)
        log(f'PatchSize:\t{self.dst["patch_size"]}', log=self.logfile, notime=True)
        log(f'LearningRate:\t{self.hyper["learning_rate"]}', log=self.logfile, notime=True)
        log(f'Epoch:\t\t{self.hyper["stop_epoch"]}', log=self.logfile, notime=True)
        log(f'num_workers:\t{self.args["num_workers"]}', log=self.logfile, notime=True)
        log(f'Command:\t{self.dst["command"]}', log=self.logfile, notime=True)
        log(f"Let's use {torch.cuda.device_count()} GPUs!", log=self.logfile, notime=True)
        # self.device != torch.device(type='cpu') 
        if 'gpu_preprocess' in self.dst and self.dst['gpu_preprocess']:
            log("Using PyTorch's GPU Preprocess...")
            self.use_gpu = True
        else:
            log(f"Using Numpy's CPU Preprocess")
            self.use_gpu = False 

        if torch.cuda.device_count() > 1:
            log("Using PyTorch's nn.DataParallel for multi-gpu...")
            self.multi_gpu = True
            self.net = nn.DataParallel(self.net)
        else:
            self.multi_gpu = False
    
    def metrics_reset(self):
        self.train_psnr.reset()
        self.eval_psnr.reset()
        self.eval_ssim.reset()
        self.eval_psnr_lr.reset()
        self.eval_psnr_dn.reset()
        self.eval_ssim_lr.reset()
        self.eval_ssim_dn.reset()

class LambdaScheduler(LambdaLR):
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

# WarmUpCosine (SGDR, ICLR 2017)
def get_cos_lr(step, period=1000, peak=20, lr=1e-4, ratio=0.2):
    T = step // period
    decay = 2 ** T
    step = step % period
    if step <= peak and T>0:
        mul = step / peak
    else:
        mul = (1-ratio) * (np.cos((step - peak) / (period - peak) * math.pi) * 0.5 + 0.5) + ratio
    return lr * mul / decay

def get_multistep_lr(step, period=1000, lr=1e-4, milestone=[500, 900], gamma=[0.5, 0.1], decay_base=1):
    decay = decay_base ** (step // period)
    step = step % period
    mul = 1
    for i in range(len(milestone), 0, -1):
        if step > milestone[i-1]:
            mul = gamma[i-1]
            break
    return lr * mul / decay
