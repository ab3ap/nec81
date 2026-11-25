'''Configuration parsing for the NEC-81 Python application.'''

from dataclasses import dataclass
from pathlib import Path

try:
    from .deck import CARD_SPECS, DeckCard, formatDeck, splitValues
except ImportError:
    from deck import CARD_SPECS, DeckCard, formatDeck, splitValues


CONFIG_TEMPLATE = '''\
# NEC-81 Python configuration
# Each setting is written as: variable value
# Relative paths are resolved from this file's directory.

# Choose one input style. Either name an existing deck:
# inputFile antenna.nec
# Or define the complete model with card groups, as shown below.

# Cards are ordered by the first appearance of each group name.
card.title.kind comment
card.title.text 'Example center-fed dipole'
card.commentEnd.kind commentEnd

card.driven.kind wire
card.driven.tag 1
card.driven.segments 21
card.driven.x1_m -0.25
card.driven.y1_m 0
card.driven.z1_m 0
card.driven.x2_m 0.25
card.driven.y2_m 0
card.driven.z2_m 0
card.driven.radius_m 0.001

card.geometryEnd.kind geometryEnd

card.frequency.kind frequency
card.frequency.stepType 0
card.frequency.count 1
card.frequency.start_MHz 299.792458

card.source.kind excitation
card.source.type 0
card.source.tag 1
card.source.segment 11
card.source.real_V 1
card.source.imag_V 0

card.pattern.kind radiationPattern
card.pattern.mode 0
card.pattern.thetaCount 19
card.pattern.phiCount 37
card.pattern.flags 1000
card.pattern.startTheta_deg 0
card.pattern.startPhi_deg 0
card.pattern.thetaStep_deg 10
card.pattern.phiStep_deg 10

card.execute.kind execute
card.end.kind end

# Generated text report.
outputFile antenna.out

# Master switch for all 3D plot generation.
plots3d true

# Individual plot categories. These require plots3d true.
plotGeometry true
plotPatterns true
plotFields true
showSources true
showLoads false

# Saved plot location and Matplotlib output format.
plotDirectory plots
plotFormat png

# Keep Matplotlib windows open for rotation, zooming, and inspection.
showPlots true
'''


class ConfigError(ValueError):
    '''Report an invalid configuration line or value.'''


@dataclass
class RunConfig:
    '''Runtime settings loaded from ``var val`` configuration lines.'''

    inputFile: str = ''
    outputFile: str = ''
    adfFile: str = ''
    writeAdf: bool = False
    plots3d: bool = False
    plotGeometry: bool = True
    plotPatterns: bool = True
    plotFields: bool = True
    showSources: bool = True
    showLoads: bool = False
    plotDirectory: str = 'plots'
    plotFormat: str = 'png'
    showPlots: bool = True
    embeddedDeck: str = ''
    somnecFile: str = ''


CONFIG_KEYS = {
    "inputFile",
    "outputFile",
    "adfFile",
    "plots3d",
    "plotGeometry",
    "plotPatterns",
    "plotFields",
    "plotDirectory",
    "plotFormat",
    "showPlots",
    "showSources",
    "showLoads",
    "writeAdf",
    'somnecFile',
}


def looksLikeConfig(path):
    """Return True if path contains NEC-81 configuration directives."""
    try:
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for rawLine in handle:
                line = rawLine.strip()
                if not line or line.startswith("#"):
                    continue
                firstWord = line.split(maxsplit=1)[0]
                return firstWord.startswith("card.") or firstWord in CONFIG_KEYS
    except (OSError, UnicodeDecodeError):
        pass
    return False


def parseBool(value, lineNumber):
    '''Convert a common text boolean to ``bool``.'''

    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    message = f'line {lineNumber}: expected a boolean, got {value!r}'
    raise ConfigError(message)


def cleanValue(value):
    '''Remove one matching pair of quotes from a value.'''

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1]:
        if value[0] in {'"', "'"}:
            return value[1:-1]
    return value


def normalizeCardName(name):
    '''Normalize descriptive card and field names for matching.'''

    return ''.join(character.lower() for character in name
                   if character.isalnum())


CARD_ALIASES = {}
for cardCode, cardSpec in CARD_SPECS.items():
    CARD_ALIASES[normalizeCardName(cardCode)] = cardCode
    CARD_ALIASES[normalizeCardName(cardSpec.name)] = cardCode


def addCardSetting(cardGroups, name, value, lineNumber):
    '''Collect one ``card.group.field value`` setting.'''

    parts = name.split('.')
    if len(parts) != 3 or not parts[1] or not parts[2]:
        message = (
            f'line {lineNumber}: expected card.group.field, got {name!r}'
        )
        raise ConfigError(message)
    groupName = parts[1]
    fieldName = normalizeCardName(parts[2])
    group = cardGroups.setdefault(groupName, {})
    if fieldName in group:
        previousLine = group[fieldName][2]
        message = (
            f'line {lineNumber}: duplicate {name!r}; first set on line '
            f'{previousLine}'
        )
        raise ConfigError(message)
    group[fieldName] = (parts[2], value, lineNumber)


def resolveAdfPath(config):
    '''Return explicit adfFile path, or <outputFile>.adf if writeAdf is set.'''

    if config.adfFile:
        return config.adfFile
    if config.writeAdf:
        base = config.outputFile or config.inputFile or 'nec81'
        return str(Path(base).with_suffix('.adf'))
    return None


def resolveCardCode(kind, lineNumber):
    '''Resolve a card code or descriptive card type name.'''

    normalized = normalizeCardName(kind)
    if normalized in CARD_ALIASES:
        return CARD_ALIASES[normalized]
    code = kind.upper().strip()
    if len(code) == 2 and code.isalpha():
        return code
    message = f'line {lineNumber}: unknown card kind {kind!r}'
    raise ConfigError(message)


def buildEmbeddedDeck(cardGroups):
    '''Build NEC deck text from collected named card groups.'''

    cards = []
    for groupName, settings in cardGroups.items():
        if 'kind' not in settings:
            firstLine = min(item[2] for item in settings.values())
            message = (
                f'line {firstLine}: card group {groupName!r} needs a '
                'kind setting'
            )
            raise ConfigError(message)
        _, kind, kindLine = settings['kind']
        code = resolveCardCode(kind, kindLine)
        spec = CARD_SPECS.get(code)
        namedFields = {
            name: item for name, item in settings.items()
            if name not in {'kind', 'values'}
        }
        if 'values' in settings:
            _, rawValues, valuesLine = settings['values']
            if namedFields:
                message = (
                    f'line {valuesLine}: card group {groupName!r} cannot '
                    'mix values with named fields'
                )
                raise ConfigError(message)
            values = splitValues(code, rawValues)
        elif spec is None:
            message = (
                f'line {kindLine}: unknown card {code} requires a values '
                'setting'
            )
            raise ConfigError(message)
        else:
            fieldIndexes = {
                normalizeCardName(fieldName): index
                for index, fieldName in enumerate(spec.fields)
            }
            values = ['']*len(spec.fields)
            for fieldName, (_, value, lineNumber) in namedFields.items():
                if fieldName not in fieldIndexes:
                    message = (
                        f'line {lineNumber}: unknown field for {kind!r}: '
                        f'{fieldName!r}'
                    )
                    raise ConfigError(message)
                values[fieldIndexes[fieldName]] = value
            while values and values[-1] == '':
                values.pop()
        cards.append(DeckCard(code, values))
    return formatDeck(cards)


def loadConfig(configPath):
    '''Load a config file whose non-comment lines are ``var val``.'''

    config = RunConfig()
    cardGroups = {}
    configPath = Path(configPath)
    fieldTypes = {
        'inputFile': str,
        'outputFile': str,
        'adfFile': str,
        'somnecFile': str,
        'plots3d': bool,
        'plotGeometry': bool,
        'plotPatterns': bool,
        'plotFields': bool,
        'showSources': bool,
        'showLoads': bool,
        'plotDirectory': str,
        'plotFormat': str,
        'showPlots': bool,
        'writeAdf': bool
    }

    with configPath.open('r', encoding='utf-8-sig') as configFile:
        for lineNumber, rawLine in enumerate(configFile, start=1):
            line = rawLine.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                message = (f'line {lineNumber}: expected "var val", got '
                           f'{line!r}')
                raise ConfigError(message)
            name, value = parts
            value = cleanValue(value)
            if name.startswith('card.'):
                addCardSetting(cardGroups, name, value, lineNumber)
                continue
            if name not in fieldTypes:
                message = f'line {lineNumber}: unknown variable {name!r}'
                raise ConfigError(message)
            if fieldTypes[name] is bool:
                value = parseBool(value, lineNumber)
            setattr(config, name, value)
    if cardGroups:
        if config.inputFile:
            message = 'choose either inputFile or embedded card groups'
            raise ConfigError(message)
        config.embeddedDeck = buildEmbeddedDeck(cardGroups)
    configDirectory = configPath.resolve().parent
    for name in ('inputFile', 'outputFile', 'adfFile', 'somnecFile',
        'plotDirectory'):
        value = getattr(config, name)
        if value and not Path(value).is_absolute():
            setattr(config, name, str(configDirectory/value))
    return config


def writeConfigTemplate(configPath):
    '''Write a commented config template without replacing an existing file.'''

    configPath = Path(configPath)
    if configPath.exists():
        message = f'refusing to overwrite existing file: {configPath}'
        raise ConfigError(message)
    configPath.parent.mkdir(parents=True, exist_ok=True)
    configPath.write_text(CONFIG_TEMPLATE, encoding='utf-8')
    return configPath


def mergeConfig(config,**overrides):
    '''Apply command-line values that are not ``None``.'''

    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    return config
