import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.store = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.store.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_all(self):
        return self.store

def PlotLearningCurve(train_loss, valid_loss, corr, epoch):
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    line1 = ax1.plot(np.arange(1,epoch+1), train_loss, '-', color='b', label='train-loss')
    line2 = ax1.plot(np.arange(1,epoch+1), valid_loss, '-', color='r', label='valid-loss')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Correction(%)')
    line3 = ax2.plot(np.arange(1,epoch+1), corr, '-', color='g', label='valid-corr')
    # legend
    line = line1 + line2 + line3
    labs = [l.get_label() for l in line]
    ax1.legend(line, labs, loc='best')

    plt.savefig('learning_curve.png')
    plt.close()
