import os
from typing import List
from src.tui.textColors import TextColors


class TUICore:
    THICK_SEPARATOR = "="
    THIN_SEPARATOR = "-"

    def __init__(self, useColor: bool = True, baseColor: str = TextColors.WHITE, baseBGColor: str = TextColors.BGWHITE):
        self.useColor = useColor
        self.baseColor = baseColor
        self.baseBGColor = baseBGColor

    def create_separator(self, text: str, thick: bool = True) -> str:
        separator = ""
        for _ in text:
            if thick:
                separator += self.THICK_SEPARATOR
            else:
                separator += self.THIN_SEPARATOR
        return separator

    def create_header(self, text: str) -> List[str]:
        separator = self.create_separator(text, thick=True)
        styled_text = f"{separator}\n{text}\n{separator}\n"
        if self.useColor:
            styled_text = TextColors.color_text(styled_text, self.baseColor, TextColors.BOLD)
        return styled_text
    
    def create_header2(self, text: str) -> List[str]:
        separator = self.create_separator(text, thick=False)
        styled_text = f"{separator}\n{text}\n{separator}\n"
        if self.useColor:
            styled_text = TextColors.color_text(styled_text, self.baseColor, TextColors.BOLD)
        return styled_text
    
    def create_message(self, text: str, *colors) -> str:
        styled_text = text
        if self.useColor:
            styled_text = TextColors.color_text(text, *colors)
        return styled_text

    def create_active_option(self, text: str) -> str:
        styled_text = f"> {text}"
        if self.useColor:
            styled_text = TextColors.color_text(text, self.baseBGColor, TextColors.BOLD)
        return styled_text

    def create_inactive_option(self, text: str) -> str:
        styled_text = f"  {text}"
        if self.useColor:
            styled_text = TextColors.color_text(text, self.baseColor)
        return styled_text

    @staticmethod
    def clear_terminal():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_buffer(terminal_buffer: List[str]):
        for line in terminal_buffer:
            print(line)

    @staticmethod
    def clear_terminal_retain_buffer(self, terminal_buffer: List[str]):
        TUICore.clear_terminal()
        for line in terminal_buffer:
            print(line)