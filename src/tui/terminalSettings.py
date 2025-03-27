import os
import sys
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty


class TerminalSettings:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        if os.name != 'nt':
            self.old_settings = termios.tcgetattr(self.fd)
        else:
            self.old_settings = None

    def __enter__(self):
        if os.name != 'nt' and self.old_settings is not None:
            tty.setcbreak(self.fd)
        else:
            # For Windows, we don't need to set terminal settings
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        if os.name != 'nt':
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        else:
            # For Windows, we don't need to restore settings
            pass