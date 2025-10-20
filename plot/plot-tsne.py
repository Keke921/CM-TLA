import os
import random
import argparse
import numpy as np
import torch
import time


from utils.config import _C as cfg
from utils.logger import setup_logger, setup_code

from trainer import Trainer


def main(args):
    cfg_data_file = os.path.join("./configs/dataset", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
        cfg.output_dir += time.strftime("-%Y-%m-%d-%H-%M-%S")
    print("Output directory: {}".format(cfg.output_dir))
    #setup_logger(cfg.output_dir)
    #setup_code(cfg.output_dir)
    
    print("** Config **")
    print(cfg)
    print("************")
    
    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg)
    
    if cfg.model_dir is not None:
        trainer.load_model(cfg.model_dir)
    
    if cfg.zero_shot:
        trainer.test()
        return

    if cfg.test_train == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_train_True")]
            print("Model directory: {}".format(cfg.model_dir))

        trainer.load_model(cfg.model_dir)
        trainer.test("train")
        return

    if cfg.test_only == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_only_True")]
            print("Model directory: {}".format(cfg.model_dir))
        
        trainer.load_model(cfg.model_dir, 'epoch-best.pth.tar')
        trainer.test()
        return

    if getattr(cfg, "tsne", True):  # tsne 可视化模式
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_tsne_True")]
            print("Model directory: {}".format(cfg.model_dir))

        trainer.load_model('./','epoch-film.pth.tar')
        trainer.tsne_visualize(train=True)   # 可视化训练集
        trainer.tsne_visualize(train=False)  # 可视化测试集
        return
        
    if getattr(cfg, "tsne_confusion", False):
        trainer.load_model('./','epoch-film.pth.tar')
        trainer.tsne_with_confusion_highlight(train=True, max_classes=10, mode="top")
        #trainer.tsne_with_confusion_highlight(train=True, select_class_mode="top", top_k=10)
        trainer.tsne_with_confusion_highlight(train=False, max_classes=10, mode="top")
        #trainer.tsne_with_confusion_highlight(train=False, select_class_mode="top", top_k=10)
        return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="cifar100_ir100", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="clip_vit_b16-confusion", help="model config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
