import torch


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_label, self.size, self.path = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)
            self.next_input = self.next_input.float()  # if need
            self.next_target = self.next_target.float()  # if need
            self.next_label = self.next_label.float()  # if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        label = self.next_label
        size = self.size
        path = self.path
        self.preload()
        return input, target, label, size, path


# val
class DataPrefetcher_val(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.size, self.path = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            # self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # self.next_label = self.next_label.cuda(non_blocking=True)
            self.next_input = self.next_input.float()  # if need
            self.next_target = self.next_target.float()  # if need
            # self.next_label = self.next_label.float()  # if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        # label = self.next_label
        size = self.size
        path = self.path
        self.preload()
        return input, target, size, path