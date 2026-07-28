import logging

class MessageColor:
    RESET = "\033[0m"

    RED = "\033[31m"
    GREEN  = "\033[32m"
    BLUE   = "\033[34m"
    YELLOW = "\033[33m"
    PINK   = "\033[38;5;205m"
    ORANGE = "\033[38;5;208m"
    PURPLE = "\033[38;5;141m"
    WHITE = "\033[37m"
    BGRED = "\033[41m"


class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        "DEBUG": 'WHITE',
        "INFO":'WHITE',
        "WARNING": 'YELLOW',
        "ERROR": 'RED',
        "CRITICAL": 'BGRED',
    }

    def format(self, record):
        color = getattr(record, "color", None)
        if color is None:
            color = self.LEVEL_COLORS.get(record.levelname, None)
        msg = super().format(record)
        if color and hasattr(MessageColor, color):
            color_code = getattr(MessageColor, color) if isinstance(color, str) else color
            msg = f"{color_code}{msg}{MessageColor.RESET}"
        return msg

