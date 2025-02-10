class TextColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    LIGHT = '\033[37m'
    ITALIC = '\033[3m'
    BLACK = '\033[30m'
    WHITE = '\033[37m'

    BGHEADER = '\033[105m'
    BGOKBLUE = '\033[104m'
    BGOKCYAN = '\033[106m'
    BGGREEN = '\033[102m'
    BGWARNING = '\033[103m'
    BGFAIL = '\033[101m'
    BGLIGHT = '\033[47m'
    BGWHITE = '\033[107m'

    @staticmethod
    def color_text(text, *colors):
        color_sequence = ''.join(colors)
        return f"{color_sequence}{text}{TextColors.ENDC}"