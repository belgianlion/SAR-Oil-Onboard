import os
import sys
from typing import List

from src.tui.keyboardInputs import KeyboardInputs
from src.tui.tuiCore import TUICore

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

class SelectMenu:
    def __init__(self, options, tuiCore: TUICore, terminal_buffer: List[str] = []):
        self.terminal_buffer = terminal_buffer
        self.tuiCore = tuiCore
        self.options = options
        self.selected_index = 0

    def print_menu(self):
        TUICore.clear_terminal_retain_buffer(self, self.terminal_buffer)
        for i, option in enumerate(self.options):
            if i == self.selected_index:
                print(self.tuiCore.create_active_option(option))
            else:
                print(self.tuiCore.create_inactive_option(option))

    def run(self) -> int:
        self.print_menu()
        while True:
            key = self.get_key()
            if key == KeyboardInputs.UP:
                self.selected_index = (self.selected_index - 1) % len(self.options)
            elif key == KeyboardInputs.DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.options)
            elif key == KeyboardInputs.ENTER:
                return self.selected_index
            self.print_menu()

    def get_key(self) -> KeyboardInputs:
        if os.name == 'nt':
            while True:
                key = msvcrt.getch()
                if key == b'\xe0':  # Special keys (arrows, f keys, ins, del, etc.)
                    key = msvcrt.getch()
                    if key == b'H':  # Up arrow
                        return KeyboardInputs.UP
                    elif key == b'P':  # Down arrow
                        return KeyboardInputs.DOWN
                elif key == b'\r':  # Enter key
                    return KeyboardInputs.ENTER
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                key = sys.stdin.read(3)
                if key == '\x1b[A':  # Up arrow
                    return KeyboardInputs.UP
                elif key == '\x1b[B':  # Down arrow
                    return KeyboardInputs.DOWN
                elif key == '\n':  # Enter key
                    return KeyboardInputs.ENTER
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)