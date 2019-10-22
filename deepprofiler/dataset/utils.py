import logging
import multiprocessing
import os
import sys
import time

PI = 3.1415926539


# From: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# Print iterations progress
def print_progress(iteration, total, prefix="Progress", suffix="Complete", decimals=1, barLength=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    if all(t >= 0 for t in [iteration, total, barLength]) and iteration <= total:
        format_str = "{0:." + str(decimals) + "f}"
        percents = format_str.format(100 * (iteration / float(total)))
        filled_length = int(round(barLength * iteration / float(total)))
        bar = "#" * filled_length + "-" * (barLength - filled_length)
        sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    elif sum([iteration < 0, total < 0, barLength < 0]) > 1:
        sys.stdout.write("\rError: print_progress() function received multiple negative values.")
        sys.stdout.flush()
    elif iteration < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'iteration' value.")
        sys.stdout.flush()
    elif total < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'total' value.")
        sys.stdout.flush()
    elif barLength < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'barLength' value.")
        sys.stdout.flush()
    elif iteration > total:
        sys.stdout.write(
            "\rError: print_progress() function received an 'iteration' value greater than the 'total' value.")
        sys.stdout.flush()


# Make sure directory exist for storing a file

def check_path(filename):
    path = "/".join(filename.split("/")[0:-1])
    os.system("mkdir -p " + path)


# Timing utilities

def tic():
    return time.time()


def toc(msg, beginning):
    end = time.time()
    elapsed = end - beginning
    print(msg, ": {:.2f} secs".format(elapsed))
    return end


# Parallel utilities using multiprocessing

class Parallel:

    def __init__(self, fixed_args, num_procs=None):
        self.fixed_args = fixed_args
        cpus = multiprocessing.cpu_count()
        if num_procs is None or num_procs > cpus or num_procs < 1:
            num_procs = cpus
        self.pool = multiprocessing.Pool(num_procs)

    def compute(self, operation, data):
        iterable = [[d, self.fixed_args] for d in data]
        self.pool.map(operation, iterable)
        return


# Logging utilities

class Logger:

    def __init__(self):
        self.root = logging.getLogger()
        self.root.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        self.root.addHandler(ch)

    def log(self, level, msg):
        self.root.log(level, msg)

    def info(self, msg):
        self.root.info(msg)


logger = Logger()
