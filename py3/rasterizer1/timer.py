import time


class Timer:
    def __init__(self, print_line="{}", verbose=True):
        self.print_line = print_line
        self.verbose = verbose

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(self.print_line.format(self.elapsed))
