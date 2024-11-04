import argparse
# Training settings
parser = argparse.ArgumentParser(description='main implementation')

#########################
#### data parameters ####
# main is my net name #
#########################
parser.add_argument("--data_name", type=str, default="mscoco_fea", # mirflickr25k mirflickr25k_fea mscoco_fea nus_wide_tc10_fea iapr_fea
                    help="data name")
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--log_name', type=str, default='main')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='main')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', help='model architecture: ' + ' | '.join(['ResNet', 'VGG']) + ' (default: vgg11)')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-6)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=256)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--num_hiden_layers', default=[3, 2], nargs='+', help='<Required> Number of hiden lyaers')
parser.add_argument('--ls', type=str, default='linear', help='lr scheduler')
parser.add_argument('--bit', type=int, default=16, help='output shape')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--momentum', type=float, default=0.999)
# 16384 65536
parser.add_argument('--K', type=int, default=65536)
parser.add_argument('--T', type=float, default=0.07)
parser.add_argument('--shift', type=float, default=0.1)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--warmup_epoch', type=int, default=1)

args = parser.parse_args()
args.num_hiden_layers = [int(i) for i in args.num_hiden_layers]
# print(args)
