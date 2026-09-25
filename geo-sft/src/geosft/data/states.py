"""Australian state and territory names -> ASUD jurisdiction codes.

Codes are ASUD's own (NSW, QLD, ...), so labels, curated metadata and the
graph lab all speak one vocabulary. OFF (offshore) has no name a passage would
use, so it is curated-only and never extracted from text.
"""
import re

STATES = {
    "new south wales": "NSW", "queensland": "QLD", "victoria": "VIC",
    "tasmania": "TAS", "south australia": "SA", "western australia": "WA",
    "northern territory": "NT", "australian capital territory": "ACT",
    "australian antarctic territory": "ATA", "antarctica": "ATA",
}

#: Full names for display (the graph lab's State nodes, benchmark questions).
STATE_NAMES = {
    "NSW": "New South Wales", "QLD": "Queensland", "VIC": "Victoria", "TAS": "Tasmania",
    "SA": "South Australia", "WA": "Western Australia", "NT": "Northern Territory",
    "ACT": "Australian Capital Territory", "ATA": "Australian Antarctic Territory",
    "OFF": "Offshore Australia",
}

#: Place names that contain a state name but are not the state. Roughly half of
#: all "Victoria" mentions in ASUD prose are the Victoria River (NT) or the
#: Great Victoria Desert (WA/SA). The gazetteer maps these to "" (longest match
#: wins, so "Victoria River" beats "Victoria") and the labeller drops them.
NOT_STATES = {
    "victoria river": "", "victoria river downs": "", "victoria river basin": "",
    "great victoria desert": "", "victoria desert": "", "lake victoria": "",
    "victoria land": "", "port victoria": "", "mount victoria": "", "mt victoria": "",
    "victoria range": "", "victoria point": "", "victoria basin": "",
    "queensland plateau": "", "queensland trough": "",
}

#: Abbreviations are only trusted in capitals -- lowercase "act" and "sa" are
#: ordinary words -- so they are matched case-sensitively, outside the gazetteer.
ABBREVIATIONS = re.compile(r"\b(NSW|QLD|Qld|VIC|Vic|TAS|Tas|SA|WA|NT|ACT)\b")
_ABBR_CODE = {"Qld": "QLD", "Vic": "VIC", "Tas": "TAS"}


def abbreviations(text: str) -> set[str]:
    return {_ABBR_CODE.get(m.group(1), m.group(1)) for m in ABBREVIATIONS.finditer(text)}
