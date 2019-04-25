from sacred import Experiment
from sacred.stflow import LogFileWriter


ex = Experiment('custom_command')

@ex.command
def scream():
    """
    scream, and shout, and let it all out ...
    """
    print('AAAaaaaaaaahhhhhh...')

# ...

@ex.capture
def some_function(_run):
    _run.result = 45

@ex.automain
@LogFileWriter(ex)
def main():
    print 'HI!!!!1'
    return 42
