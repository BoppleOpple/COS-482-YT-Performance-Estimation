# region Partial ANSI Table
ansiCodes = {
    "normal": "0",
    "bold": "1",
    "faint": "2",
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "bright_black": "90",
    "bright_red": "91",
    "bright_green": "92",
    "bright_yellow": "93",
    "bright_blue": "94",
    "bright_magenta": "95",
    "bright_cyan": "96",
    "bright_white": "97",
}
# endregion


# region getANSI
def getANSI(*args):
    ansi = ";".join([ansiCodes[arg] for arg in args])
    return f"\033[{ansi}m"


# endregion


# region resetANSI
def resetANSI():
    return getANSI("black", "normal")


# endregion


# region printANSI
def printANSI(string, *args):
    print(f"{getANSI(*args)}{string}", resetANSI())


# endregion


# region printBox
def printBox(string, *args):
    print(getANSI(*args))
    print("+-" + "".join(["-" for c in string]) + "-+")
    print("| " + "".join([" " for c in string]) + " |")
    print("| " + string + " |")
    print("| " + "".join([" " for c in string]) + " |")
    print("+-" + "".join(["-" for c in string]) + "-+")
    print(resetANSI())


# endregion
