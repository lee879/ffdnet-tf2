import argparse
import numpy as np
import time

def easy_arg():
    parser = argparse.ArgumentParser()
    # trian
    parser.add_argument("--train-path", type=str, default="./train_data",                   help="Train data dir")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt",                          help="Get the ckpt save path")
    parser.add_argument("--batch", type=int, default=1,                                     help="Train batch default 32 ")
    parser.add_argument("--patch_size_batch", type=int, default=1024,                       help="patch_size_batch")
    parser.add_argument("--patch_size",type=int, default=1024,                               help="PatchSize Setting")
    parser.add_argument("--train_noise_intercal", nargs=3, type=int, default=[0,20,2],     help="Train dataset noise sigma set interval")
    parser.add_argument("--val_train_noise_intercal", nargs=3, type=int, default=[0,60,10], help="Validation dataset noise sigma set interval")
    parser.add_argument("--epochs", type=int, default=30000000000,                          help="All epoch is this")
    parser.add_argument("--lr", type=float, default=0.00001,                               help="Learning-rate setting")
    parser.add_argument("--optim", type=int, default=0,                                     help=("Setting Adam(0) or SGD(1) Opitm"))
    parser.add_argument("--ckpt_epoch", type=int, default=100,                              help="Save checkpoint every epoch")
    parser.add_argument("--cos_lr", type=int, default=0,                                    help="Setting 0 to use cos_lr")
    parser.add_argument("--layers", type=int, default=12,                                   help="Conv layers")
    parser.add_argument("--log", type=str, default="./log",                                 help="Tensorboard logs")
    parser.add_argument("--img_size", type=int, default=1024,                               help="Image size")
    # Global
    parser.add_argument("--GPU", action='store_true',default=True,                          help="To use GPU")
    parser.add_argument("--model-path", type=str, default="./model",                        help="Model save path")
    parser.add_argument("--is_train", action='store_true', default=True,                    help='Do train')
    parser.add_argument("--is_test", action='store_true', default=True,                     help='Do test')
    parser.add_argument("--gray_rgb", type=int, default=1,                                  help="Setting 0 to select gray")
    parser.add_argument("--noise_type",type=int, default=0,                                 help="Gaussian(0),Salt_and_Pepper(1),Speckle(2)")
    parser.add_argument("--image_size",type=int, default=256,                               help="Init image size")
    return parser.parse_args()

def args_infom():
    args = easy_arg()
    assert (args.is_train or args.is_test),\
        'is_train 和 is_test 至少有一个为 True'
    data_dict = {}

    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        data_dict[k] = v
    # 创建 JavaScript 代码字符串
    js_code = "var data = " + str(data_dict)

    #写入文件,文件名使用时间戳加上一个随机数
    with open("./parser/{}.js".format(str(int(time.time()) + np.random.randint(low=0,high=100,size=1))), "w") as f:
        f.write(js_code)
    #打印相应的参数
    print("----------------------------------All Parser----------------------------------")
    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print(f'\t{k}: {v}')
    print("----------------------------------end----------------------------------")
    return None
if __name__ == '__main__':
    args_infom()


































