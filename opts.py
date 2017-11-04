import argparse

optim_choices = ['sgd','adam','adagrad', 'adamax', 'adadelta']

def myargparser():
    parser = argparse.ArgumentParser(description='GazeNet Training')

    #data stuff
    parser.add_argument('--dataset', default='gazefollow', type=str, help='chosen dataset')
    parser.add_argument('--data_dir', required=True, type=str, help='chosen data directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    #default stuff
    parser.add_argument('--epochs', required=True, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', required=True, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--testbatchsize', required=True, type=int, help='input batch size for testing (default: 1000)')
    parser.add_argument('--printfreq', default=200, type=int, help='print frequency (default: 10)')
    parser.add_argument('--learningratescheduler', default='decayschedular', type=str, help='if lr rate scheduler should be used')

    #optimizer/criterion stuff
    parser.add_argument('--decayinterval', default=50, type=int, help='decays by a power of decay_var in these epochs')
    parser.add_argument('--decaylevel', default=2, type=int, help='decays by a power of decaylevel')
    parser.add_argument('--criterion', default='mse', help='Criterion')
    parser.add_argument('--optimType', default='adam', choices=optim_choices, type=str, help='Optimizers. Options:'+str(optim_choices))

    parser.add_argument('--maxlr', default=0.001, required=True, type=float, help='initial learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--minlr', default=0.0005, required=True, type=float, help='initial learning rate')

    parser.add_argument('--nesterov', action='store_true', help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (Default: 0.9)')
    parser.add_argument('--weightDecay', default=0, type=float, help='weight decay (Default: 1e-4)')

    #extra model stuff
    parser.add_argument('--model_def', default='gazenet', help='Architectures to be loaded')
    parser.add_argument('--inpsize', default=227, type=int, help='Input size')
    parser.add_argument('--weight_init', action='store_false', help='Turns off weight inits')

    #default
    parser.add_argument('--cachemode', default=True, help='if cachemode')
    parser.add_argument('--cuda',  default=True, help='if cuda is available')
    parser.add_argument('--manualSeed',  default=123, help='fixed seed for experiments')
    parser.add_argument('--testOnly', default=False, type=bool, help='run on validation set only')
    parser.add_argument('--start-epoch', default=0, type=int,help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--pretrained_file', default="")

    #model stuff
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--store', default='', type=str,
                        help='path to storing checkpoints (default: none)')


    return parser
