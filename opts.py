import argparse

optim_choices = ['sgd','adam']

def optionargparser():
    parser = argparse.ArgumentParser(description='GazeNet Training')

    #data stuff
    parser.add_argument('--dataset', default='gazefollow', type=str, help='chosen dataset')
    parser.add_argument('--data_dir', default='../data/', type=str, help='chosen data directory')
    parser.add_argument('--placesmodelpath', default='../whole_alexnet_places365.pth.tar', type=str, help='chosen data directory')
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    #default stuff
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=50, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--testbatchsize', default=128, type=int, help='input batch size for testing (default: 1000)')
    parser.add_argument('--printfreq', default=50, type=int, help='print frequency (default: 10)')
    parser.add_argument('--learningratescheduler', default='decayschedular', type=str, help='if lr rate scheduler should be used')

    #optimizer/criterion stuff
    parser.add_argument('--decayinterval', default=10, type=int, help='decays by a power of decay_var in these epochs')
    parser.add_argument('--decaylevel', default=1.15, type=int, help='decays by a power of decaylevel')
    parser.add_argument('--criterion', default='crossentropy', help='Criterion')
    parser.add_argument('--optimType', default='adam', choices=optim_choices, type=str, help='Optimizers. Options:'+str(optim_choices))

    parser.add_argument('--maxlr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--minlr', default=0.00001, type=float, help='initial learning rate')

    parser.add_argument('--nesterov', action='store_true', help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (Default: 0.9)')
    parser.add_argument('--weightDecay', default=1e-6, type=float, help='weight decay (Default: 1e-4)')

    #extra model stuff
    parser.add_argument('--inpsize', default=227, type=int, help='Input size')
    parser.add_argument('--shiftedflag', default=True, help='whether shifted grids is turned on')

    #default
    parser.add_argument('--seed',  default=123, help='fixed seed for experiments')
    parser.add_argument('--testOnly', default=False, type=bool, help='run on validation set only')
    parser.add_argument('--start-epoch', default=0, type=int,help='manual epoch number (useful on restarts)')

    #model stuff
    parser.add_argument('--resume', default='none', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--store', default='', type=str,
                        help='path to storing checkpoints (default: none)')


    return parser
