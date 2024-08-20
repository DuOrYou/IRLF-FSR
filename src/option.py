import argparse
from multiprocessing.dummy import Pool

from numpy import array
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='Our_HAT_woF',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

parser.add_argument('--gpunum', type=str, default='0',
                    help='random seed')

# Data specifications
parser.add_argument('--original_data', type=str, default='.dataaset/CleanIMG',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='.dataaset/demo_img',
                    help='demo image directory')


parser.add_argument('--train_dir', type=str, default='.dataase/Hybrid_train',
                    help='train dataset directory')

parser.add_argument('--test_only_dir', type=str, default='.dataaset/test_img',
                    help='test only  directory')
parser.add_argument('--video_dir', type=str, default='.dataaset/real_image',
                    help='test real-world image')
# parser.add_argument('--gen_train_data',action='store_true',
#                     help='train dataset directory')
parser.add_argument('--gen_train_data', type=str, default=False,
                    help='if image is gray or not')

parser.add_argument('--without_gt', type=str, default=False,
                    help='if image is gray or not')

parser.add_argument('--data_augment',action='store_true',
                    help='add data augment')

parser.add_argument('--val_dir', type=str, default='.dataaset/val_data',
                    help='val dataset directory')
parser.add_argument('--test_dir', type=str, default='.dataaset/test_data',
                    help='test dataset directory')


parser.add_argument('--test_unknown',action='store_true',
                    help='test unknown data')


parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')

parser.add_argument('--noise_level', type=list, default=[0,0.01,0.03,0.05],
                    help='noise_level of validation set, when testing real-world image set this to None')


parser.add_argument('--crop_times', type=int, default=16,
                    help='crop times per image, only enabled when trying to generta train data ')
parser.add_argument('--patch_size', type=int, default=32,
                    help='lr img patch size')
parser.add_argument('--angular', type=int, default=3,
                    help='angular resolution of light field')
parser.add_argument('--is_gray_scale', type=str, default=True,
                    help='if image is gray or not')
parser.add_argument('--test_with_patchs', type=str, default=False,
                    help='True for test with patches')

parser.add_argument('--nonuniform_N', type=str, default=True,
                    help='set the type of noise type')
parser.add_argument('--test_via_patches', type=str, default=True,
                    help='if image is gray or not')

parser.add_argument('--defaultname', type=str, default='',
                    help='if test selected img or not')


parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='None',
                    help='model name')

parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=32,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--train_uv', type=str, default='',
                    help='number of feature maps')

parser.add_argument('--is_sisr', type=str, default=False,
                    help='if or not sisr method')

parser.add_argument('--is_misr', type=str, default=False,
                    help='if or not misr method')
parser.add_argument('--KL_loss', type=str, default=False,
                    help='if or not misr method')
# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--groups', type=int, default=1,
                    help='number of groups')

parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')

parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--bytedepth', type=int, default=8,
                    help='byte depth of pixel')
parser.add_argument('--decay_every', type=int, default=15,
                    help='lr decay rate')

parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')

parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--layer_num', type=str, default='6',
                    help='learning rate')
parser.add_argument('--decay', type=str, default='10000',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')

# parser.add_argument('--KL_loss', action='store_true',
#                     help='KL_loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications

parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

parser.add_argument('--data_test', type=str, default='AISRs',
                    help='test dataset directory')

args = parser.parse_args()
template.set_template(args)

# args.scale = list(map(lambda x: int(x), args.scale.split('+')))
# args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')
# args.save_results = True
# args.save_models = True
if args.epochs == 0:
    args.epochs = 1e8
args.gen_train_data = 'False'
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

