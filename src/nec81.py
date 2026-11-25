#!/usr/bin/env python

'''Command-line interface for the Python conversion of NEC-81.'''

import argparse
from pathlib import Path
import sys
import tempfile

packageDirectory = str(Path(__file__).resolve().parent)
if packageDirectory not in sys.path:
    sys.path.insert(0, packageDirectory)

from .config import (
    ConfigError,
    RunConfig,
    loadConfig,
    looksLikeConfig,
    mergeConfig,
    writeConfigTemplate,
)
from .plots3d import configurePlots
from .nsmaExport import configureAdfExport


def buildParser():
    '''Build the NEC-81 command-line parser.'''

    parser = argparse.ArgumentParser(
        description='Run the Python conversion of NEC-81.', )
    parser.add_argument('inputFile', nargs='?', help='NEC input deck')
    parser.add_argument('outputFile', nargs='?', help='report output file')
    parser.add_argument(
        '-a', '--adf-file',
        dest='adfFile',
        nargs='?',
        const=True,
        default=None,
        metavar='PATH',
        help='write far-field pattern to NSMA .adf file',
    )
    parser.add_argument(
        '-c', '--config',
        dest='configFile',
        help='configuration file containing "var val" lines',
    )
    parser.add_argument(
        '-g', '--gui',
        nargs='?',
        const='',
        dest='guiDeck',
        metavar='DECK',
        help='open the desktop interface, optionally with a NEC deck',
    )
    parser.add_argument(
        '-o', '--cli', '--no-gui',
        action='store_true',
        dest='cliMode',
        help='run in original command-line mode',
    )
    parser.add_argument(
        '-s', '--somnec-file',
        dest='somnecFile',
        help='Sommerfeld-Norton grid file (fort.21) for GN 2 ground',
    )
    parser.add_argument(
        '-t', '--write-config-template',
        nargs='?',
        const='nec81.cfg',
        dest='configTemplate',
        metavar='PATH',
        help='write a commented config template and exit',
    )
    plotGroup = parser.add_mutually_exclusive_group()
    plotGroup.add_argument(
        '--plots3d',
        action='store_true',
        dest='plots3d',
        default=None,
        help='write 3D plot images',
    )
    plotGroup.add_argument(
        '--no-plots3d',
        action='store_false',
        dest='plots3d',
        help='disable 3D plot images',
    )
    parser.add_argument(
        '-d', '--plot-directory',
        dest='plotDirectory',
        help='directory for plot images',
    )
    showGroup = parser.add_mutually_exclusive_group()
    showGroup.add_argument(
        '--show-plots',
        action='store_true',
        dest='showPlots',
        default=None,
        help='keep interactive plot windows open',
    )
    showGroup.add_argument(
        '--no-show-plots',
        action='store_false',
        dest='showPlots',
        help='save plot images without opening windows',
    )
    loadsGroup = parser.add_mutually_exclusive_group()
    loadsGroup.add_argument(
        '--show-loads',
        action='store_true',
        dest='showLoads',
        default=None,
        help='show impedance load markers on 3D geometry plots',
    )
    loadsGroup.add_argument(
        '--no-show-loads',
        action='store_false',
        dest='showLoads',
        help='hide impedance load markers on 3D geometry plots',
    )
    sourcesGroup = parser.add_mutually_exclusive_group()
    sourcesGroup.add_argument(
        '--show-sources',
        action='store_true',
        dest='showSources',
        default=None,
        help='show excitation source markers on 3D geometry plots',
    )
    sourcesGroup.add_argument(
        '--no-show-sources',
        action='store_false',
        dest='showSources',
        help='hide excitation source markers on 3D geometry plots',
    )
    return parser

def resolveConfig(arguments):
    """Merge configuration-file and command-line settings."""

    adfFileArg = arguments.adfFile if isinstance(arguments.adfFile, str) \
        else None
    writeAdfArg = True if arguments.adfFile is True else None
    if arguments.configFile:
        config = loadConfig(arguments.configFile)
    elif arguments.inputFile and looksLikeConfig(arguments.inputFile):
        config = loadConfig(arguments.inputFile)
        arguments.inputFile = None
    else:
        config = RunConfig()
    config = mergeConfig(
        config,
        inputFile=arguments.inputFile,
        outputFile=arguments.outputFile,
        somnecFile=arguments.somnecFile,
        plots3d=arguments.plots3d,
        plotDirectory=arguments.plotDirectory,
        showPlots=arguments.showPlots,
        adfFile=adfFileArg,
        writeAdf=writeAdfArg,
    )
    if arguments.inputFile is not None:
        config.embeddedDeck = ""
    return config


def runFromConfig(config):
    '''Configure optional outputs and execute NEC-81.'''

    configurePlots(
        enabled=config.plots3d,
        geometryEnabled=config.plotGeometry,
        patternsEnabled=config.plotPatterns,
        fieldsEnabled=config.plotFields,
        showSources=config.showSources,
        showLoads=config.showLoads,
        outputDirectory=config.plotDirectory,
        imageFormat=config.plotFormat,
        showPlots=config.showPlots,
    )
    from .mainsa import runNec81
    from .config import resolveAdfPath
    configureAdfExport(outputPath=resolveAdfPath(config))

    temporaryDeck = None
    inputFile = config.inputFile or None
    if config.embeddedDeck:
        deckFile = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.nec',
            prefix='nec81_config_',
            delete=False,
            encoding='utf-8',
        )
        with deckFile:
            deckFile.write(config.embeddedDeck)
        temporaryDeck = Path(deckFile.name)
        inputFile = str(temporaryDeck)
    try:
        return runNec81(
            inputFile=inputFile,
            outputFile=config.outputFile or None,
            somnecFile=config.somnecFile or None,
        )
    finally:
        if temporaryDeck is not None:
            temporaryDeck.unlink(missing_ok=True)


def main(argv=None):
    '''Run the NEC-81 application (GUI by default unless CLI flags/args are
    given).'''

    rawArgs = list(sys.argv[1:] if argv is None else argv)
    parser = buildParser()
    arguments = parser.parse_args(rawArgs)

    # Determine if user explicitly requested CLI/batch execution
    wantsCli = (
        arguments.cliMode
        or arguments.configTemplate is not None
        or arguments.configFile is not None
        or arguments.outputFile is not None
        or arguments.adfFile is not None
        or arguments.plots3d is not None
        or arguments.plotDirectory is not None
        or arguments.showPlots is not None
    )

    try:
        # Default to GUI when run with no arguments or with a single
        # .nec/.cfg file.
        if not wantsCli:
            from .gui import launchGui

            initial = arguments.guiDeck if arguments.guiDeck is not None \
                else arguments.inputFile
            launchGui(initial or None)
            return 0

        if arguments.configTemplate:
            outputPath = writeConfigTemplate(arguments.configTemplate)
            print(f'Wrote configuration template: {outputPath}')
            return 0

        config = resolveConfig(arguments)
        return runFromConfig(config)
    except (ConfigError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
