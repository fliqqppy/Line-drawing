import pickle
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from dataset import ImageSet
from ecn import ECN, BasicBlock, Bottleneck
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(data_loader, losses):
    ecn.train()
    for idx, (lines, anns) in enumerate(data_loader):
        lines = lines.to(device)
        anns = anns.to(device)

        output = ecn(lines)
        loss = loss_fn(output, anns)

        # back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        show_loss = loss.cpu().data.numpy()
        losses.append(show_loss)
        print('[EPOCH] {}, batch: {}, loss: {}'.format(epoch, idx, show_loss))


def validate(data_loader, losses):
    print('testing...')
    ecn.eval()
    i = 0
    loss_sum = 0
    with torch.no_grad():
        for idx, (lines, anns) in enumerate(data_loader):
            lines = lines.to(device)
            anns = anns.to(device)

            output = ecn(lines)
            loss = loss_fn(output, anns)
            loss_sum += loss.cpu().data.numpy()
            i = idx

    avg_loss = loss_sum / (i+1)
    losses.append(avg_loss)
    print('Average loss: ' + str(avg_loss))
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spilt', type=str, default='rcf', help='which dataset to use')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--losses', type=str, default='plots/losses_total_pp.pkl',
                        help='load saved losses, use this only if continue training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=4, help='total epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--tl', type=bool, default=True, help='use transfer learning')
    parser.add_argument('--pretrained', type=str, default='ecn_tl.tar',
                        help='pre-trained model used for transfer learning')
    args = parser.parse_args()

    spilt = args.spilt

    # load model
    ecn = ECN(en_block=Bottleneck, de_block=BasicBlock, zero_init_residual=True)
    ecn.to(device)

    optimizer = optim.Adam(ecn.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    loss_fn = nn.MSELoss().to(device)

    train_set = ImageSet(root=spilt + '-data', i=3)
    test_set = ImageSet(root='test-data', i=2)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # start training
    print('Start training...')
    training_losses = []
    testing_losses = []
    best_loss = 1
    start_epoch = 1
    # resume from checkpoint
    if args.resume:
        model_file = args.model_path + '/ecn_{}.tar'.format(spilt)
        checkpoint = torch.load(model_file)
        curr_epoch = checkpoint['epoch']
        start_epoch = curr_epoch + 1
        training_losses = pickle.load(open(args.losses, 'rb'))
        ecn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('Checkpoint loaded! Start from epoch: {}'.format(start_epoch))

    if args.tl:
        # load pretrained model
        spilt = spilt + '_tl'
        pretrained = args.model_path + '/' + args.pretrained
        model = torch.load(pretrained)
        ecn.load_state_dict(model['model_state_dict'])

    for epoch in range(start_epoch, args.epoch+1):
        scheduler.step()
        train(train_loader, training_losses)
        # curr_loss = validate(test_loader, testing_losses)
        # save checkpoint
        state = {
            'epoch': epoch,
            'model_state_dict': ecn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        # if epoch % 2 == 0:
        torch.save(state, 'checkpoints/ecn_{}_ckpt_{}.tar'.format(spilt, epoch))
        print('Checkpoint saved!')
        # if curr_loss < best_loss:
        torch.save(state, args.model_path + '/ecn_{}.tar'.format(spilt))
        print('Model saved!')
        # best_loss = curr_loss

    with open('plots/losses_train_{}.pkl'.format(spilt), 'wb') as f:
        pickle.dump(training_losses, f)
    # with open('plots/losses_test_{}.pkl'.format(spilt), 'wb') as f:
    #     pickle.dump(testing_losses, f)

    print('Finished training.')



