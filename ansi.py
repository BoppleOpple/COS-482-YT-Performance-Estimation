ansiCodes = {
	"normal":         "0",
	"bold":         "1",
	"faint":         "2",
	"black":          "30",
	"red":            "31",
	"green":          "32",
	"yellow":         "33",
	"blue":           "34",
	"magenta":        "35",
	"cyan":           "36",
	"white":          "37",
	"bright_black":   "90",
	"bright_red":     "91",
	"bright_green":   "92",
	"bright_yellow":  "93",
	"bright_blue":    "94",
	"bright_magenta": "95",
	"bright_cyan":    "96",
	"bright_white":   "97"
}

def getANSI(*args):
	return f"\033[{";".join([ansiCodes[arg] for arg in args])}m"

def resetANSI():
	return getANSI("black", "normal")