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
