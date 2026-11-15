'''Qt desktop interface for editing and running NEC-81 decks.'''

import codecs
import os
from pathlib import Path
import sys
import tempfile

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
    from PySide6.QtGui import (
        QAction,
        QFontDatabase,
        QKeySequence,
        QTextCursor,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:
    raise RuntimeError(
        'The NEC-81 GUI requires PySide6. Install it with: '
        'python -m pip install PySide6'
    ) from error

# Matplotlib must be imported after PySide6 so it selects the same binding.
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure

packageDirectory = Path(__file__).resolve().parent

from .config import loadConfig, looksLikeConfig
from .deck import (
    CARD_SPECS,
    DeckCard,
    cardSummary,
    formatDeck,
    geometryBounds,
    loadDeck,
    parseDeck,
    previewModel,
    saveDeck,
    validateDeck,
)

Buttons = QMessageBox.StandardButton


def monospaceFont(size_pt):
    '''Return the system fixed-width font at the requested point size.'''

    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(size_pt)
    return font


def makeButton(text, command):
    '''Create a push button that calls command without arguments.'''

    button = QPushButton(text)
    button.clicked.connect(lambda checked=False: command())
    return button


class Nec81Gui(QMainWindow):
    '''Provide a deck editor, preview, runner, report, and plot browser.'''

    def __init__(self, initialDeck=None):
        super().__init__()
        self.resize(1280, 840)
        self.setMinimumSize(960, 640)
        self.cards = []
        self.deckPath = None
        self.dirty = False
        self.fieldEntries = []
        self.shownCode = ''
        self.process = None
        self.stopRequested = False
        self.outputDecoder = None
        self.temporaryConfig = None
        self.temporaryDeck = None
        self.plotFiles = []
        self.actions = {}
        self.buildMenu()
        self.buildToolbar()
        self.buildNotebook()
        self.setStatus('Ready')
        if initialDeck:
            if looksLikeConfig(initialDeck):
                self.openConfigPath(initialDeck)
            else:
                self.openDeckPath(initialDeck)
        else:
            self.newDeck()


    def setStatus(self, text):
        '''Show a message in the status bar.'''

        self.statusBar().showMessage(text)

    def addAction(self, menu, key, text, command, shortcut=None):
        '''Create a menu action and remember it for the toolbar.'''

        action = QAction(text, self)
        action.triggered.connect(lambda checked=False: command())
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        menu.addAction(action)
        self.actions[key] = action

    def buildMenu(self):
        '''Create the application menu bar.'''

        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        self.addAction(fileMenu, 'new', '&New Deck', self.newDeck,
                       QKeySequence.StandardKey.New)
        self.addAction(fileMenu, 'open', '&Open Deck...', self.openDeck,
                       QKeySequence.StandardKey.Open)
        self.addAction(fileMenu, 'config', 'Open &Config...',
                       self.openConfig)
        fileMenu.addSeparator()
        self.addAction(fileMenu, 'save', '&Save', self.saveDeck,
                       QKeySequence.StandardKey.Save)
        self.addAction(fileMenu, 'saveAs', 'Save &As...', self.saveDeckAs,
                       QKeySequence.StandardKey.SaveAs)
        fileMenu.addSeparator()
        self.addAction(fileMenu, 'exit', 'E&xit', self.close,
                       QKeySequence.StandardKey.Quit)
        deckMenu = menuBar.addMenu('&Deck')
        self.addAction(deckMenu, 'validate', '&Validate',
                       self.validateCurrentDeck)
        self.addAction(deckMenu, 'applyRaw', 'Apply &Raw Text',
                       self.applyRawText)
        self.addAction(deckMenu, 'preview', 'Refresh &Geometry',
                       self.refreshGeometry)
        runMenu = menuBar.addMenu('&Run')
        self.addAction(runMenu, 'run', '&Run NEC-81', self.runSolver, 'F5')
        self.addAction(runMenu, 'stop', 'S&top Run', self.stopSolver,
                       'Shift+F5')
        helpMenu = menuBar.addMenu('&Help')
        self.addAction(helpMenu, 'help', 'GUI &Help', self.showHelp,
                       QKeySequence.StandardKey.HelpContents)
        self.addAction(helpMenu, 'about', '&About', self.showAbout)

    def buildToolbar(self):
        '''Create frequently used command buttons.'''

        toolbar = QToolBar('Main', self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for key in ('new', 'open', 'save', 'validate', 'preview', 'run'):
            toolbar.addAction(self.actions[key])

    def buildNotebook(self):
        '''Create all primary workspace tabs.'''

        self.notebook = QTabWidget()
        self.setCentralWidget(self.notebook)
        self.deckTab = QWidget()
        self.rawTab = QWidget()
        self.geometryTab = QWidget()
        self.runTab = QWidget()
        self.reportTab = QWidget()
        self.plotsTab = QWidget()
        self.notebook.addTab(self.deckTab, 'Card Editor')
        self.notebook.addTab(self.rawTab, 'Raw Deck')
        self.notebook.addTab(self.geometryTab, 'Geometry')
        self.notebook.addTab(self.runTab, 'Run')
        self.notebook.addTab(self.reportTab, 'Report')
        self.notebook.addTab(self.plotsTab, 'Saved Plots')
        self.buildDeckTab()
        self.buildRawTab()
        self.buildGeometryTab()
        self.buildRunTab()
        self.buildReportTab()
        self.buildPlotsTab()

    def buildDeckTab(self):
        '''Create card table and structured card editor.'''

        layout = QHBoxLayout(self.deckTab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        self.cardTable = QTableWidget(0, 3)
        self.cardTable.setHorizontalHeaderLabels(['#', 'Card', 'Summary'])
        self.cardTable.verticalHeader().setVisible(False)
        self.cardTable.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.cardTable.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.cardTable.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        header = self.cardTable.horizontalHeader()
        header.setStretchLastSection(True)
        self.cardTable.setColumnWidth(0, 48)
        self.cardTable.setColumnWidth(1, 65)
        self.cardTable.currentCellChanged.connect(
            lambda row, column, oldRow, oldColumn: self.selectCard()
        )
        splitter.addWidget(self.cardTable)
        editor = QWidget()
        editorLayout = QVBoxLayout(editor)
        typeRow = QHBoxLayout()
        typeRow.addWidget(QLabel('Card Type'))
        self.codeCombo = QComboBox()
        self.codeCombo.setEditable(True)
        self.codeCombo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.codeCombo.addItems(sorted(CARD_SPECS))
        self.codeCombo.setMinimumWidth(90)
        self.codeCombo.activated.connect(
            lambda index: self.changeCardType()
        )
        self.codeCombo.lineEdit().editingFinished.connect(
            self.changeCardType
        )
        typeRow.addWidget(self.codeCombo)
        typeRow.addStretch()
        editorLayout.addLayout(typeRow)
        self.cardDescription = QLabel()
        self.cardDescription.setWordWrap(True)
        self.cardDescription.setEnabled(False)
        editorLayout.addWidget(self.cardDescription)
        self.fieldsForm = QFormLayout()
        editorLayout.addLayout(self.fieldsForm)
        editorLayout.addStretch()
        buttonRow = QHBoxLayout()
        buttons = (
            ('Apply', self.applyCardEdits),
            ('Insert', self.insertCard),
            ('Delete', self.deleteCard),
            ('Move Up', lambda: self.moveCard(-1)),
            ('Move Down', lambda: self.moveCard(1)),
        )
        for label, command in buttons:
            buttonRow.addWidget(makeButton(label, command))
        editorLayout.addLayout(buttonRow)
        splitter.addWidget(editor)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.rebuildFields()

    def buildRawTab(self):
        '''Create a full-text deck editor.'''

        layout = QVBoxLayout(self.rawTab)
        controls = QHBoxLayout()
        controls.addWidget(makeButton('Apply Raw Text', self.applyRawText))
        controls.addWidget(QLabel(
            'Apply changes before using the card editor or running.'
        ))
        controls.addStretch()
        layout.addLayout(controls)
        self.rawText = QPlainTextEdit()
        self.rawText.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.rawText.setFont(monospaceFont(10))
        layout.addWidget(self.rawText)

    def buildGeometryTab(self):
        '''Create an interactive embedded 3D wire preview.'''

        layout = QVBoxLayout(self.geometryTab)
        controls = QHBoxLayout()
        controls.addWidget(makeButton('Refresh Preview',
                                      self.refreshGeometry))
        controls.addWidget(QLabel(
            'Drag to rotate. Use the toolbar to pan, zoom, or save. '
            'The preview currently renders GW wire cards.'
        ))
        controls.addStretch()
        layout.addLayout(controls)
        self.geometryFigure = Figure(figsize=(8, 6), dpi=100)
        self.geometryAxis = self.geometryFigure.add_subplot(
            111, projection='3d'
        )
        self.geometryCanvas = FigureCanvasQTAgg(self.geometryFigure)
        layout.addWidget(NavigationToolbar2QT(self.geometryCanvas, self))
        layout.addWidget(self.geometryCanvas, stretch=1)

    def buildRunTab(self):
        '''Create solver and plot settings plus process output.'''

        layout = QVBoxLayout(self.runTab)
        settings = QGroupBox('Run Settings')
        grid = QGridLayout(settings)
        self.outputPath = QLineEdit()
        self.plotDirectory = QLineEdit('plots')
        self.plotFormat = QComboBox()
        self.plotFormat.setEditable(True)
        self.plotFormat.addItems(['png', 'svg', 'pdf'])

        grid.addWidget(QLabel('Report File'), 0, 0)
        grid.addWidget(self.outputPath, 0, 1)
        grid.addWidget(makeButton('Browse...', self.chooseOutputPath), 0, 2)

        grid.addWidget(QLabel('Plot Directory'), 1, 0)
        grid.addWidget(self.plotDirectory, 1, 1)
        grid.addWidget(makeButton('Browse...', self.choosePlotDirectory), 1, 2)

        grid.addWidget(QLabel('Plot Format'), 2, 0)
        grid.addWidget(self.plotFormat, 2, 1, Qt.AlignmentFlag.AlignLeft)
        checks = QHBoxLayout()
        self.plots3d = QCheckBox('Enable 3D plots')
        self.plotGeometry = QCheckBox('Geometry')
        self.plotPatterns = QCheckBox('Patterns')
        self.plotFields = QCheckBox('Fields')
        self.showSources = QCheckBox('Sources')
        self.showLoads = QCheckBox('Loads')
        self.writeAdf = QCheckBox('Write NSMA .adf')
        self.writeAdf.setChecked(False)
        self.showPlots = QCheckBox('Open interactive windows')
        for box in (self.plots3d, self.plotGeometry, self.plotPatterns,
                self.plotFields, self.showSources, self.showPlots):
            box.setChecked(True)
        self.showLoads.setChecked(False)
        for box in (self.plots3d, self.plotGeometry, self.plotPatterns,
            self.plotFields, self.showSources, self.showLoads,
            self.showPlots, self.writeAdf):
            checks.addWidget(box)
        checks.addStretch()
        grid.addLayout(checks, 3, 0, 1, 3)
        controls = QHBoxLayout()
        self.runButton = makeButton('Run NEC-81', self.runSolver)
        self.stopButton = makeButton('Stop', self.stopSolver)
        self.stopButton.setEnabled(False)
        controls.addWidget(self.runButton)
        controls.addWidget(self.stopButton)
        controls.addStretch()
        grid.addLayout(controls, 4, 0, 1, 3)
        grid.setColumnStretch(1, 1)
        layout.addWidget(settings)
        layout.addWidget(QLabel('Process Output'))
        self.processText = QPlainTextEdit()
        self.processText.setReadOnly(True)
        self.processText.setFont(monospaceFont(9))
        layout.addWidget(self.processText, stretch=1)

    def buildReportTab(self):
        '''Create the solver report viewer.'''

        layout = QVBoxLayout(self.reportTab)
        controls = QHBoxLayout()
        controls.addWidget(makeButton('Reload Report', self.loadReport))
        controls.addStretch()
        layout.addLayout(controls)
        self.reportText = QPlainTextEdit()
        self.reportText.setReadOnly(True)
        self.reportText.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.reportText.setFont(monospaceFont(9))
        layout.addWidget(self.reportText)

    def buildPlotsTab(self):
        '''Create a browser for plot images saved by the solver.'''

        layout = QVBoxLayout(self.plotsTab)
        controls = QHBoxLayout()
        controls.addWidget(makeButton('Reload Plot Files',
                                      self.loadPlotFiles))
        self.plotCombo = QComboBox()
        self.plotCombo.setMinimumWidth(320)
        self.plotCombo.activated.connect(lambda index: self.displayPlot())
        controls.addWidget(self.plotCombo)
        controls.addStretch()
        layout.addLayout(controls)
        self.plotFigure = Figure(figsize=(8, 6), dpi=100)
        self.plotAxis = self.plotFigure.add_subplot(111)
        self.plotAxis.axis('off')
        self.plotCanvas = FigureCanvasQTAgg(self.plotFigure)
        layout.addWidget(NavigationToolbar2QT(self.plotCanvas, self))
        layout.addWidget(self.plotCanvas, stretch=1)

    def setDirty(self, dirty=True):
        '''Update modified state and window title.'''

        self.dirty = dirty
        name = self.deckPath.name if self.deckPath else 'Untitled.nec'
        self.setWindowTitle(f'NEC-81 Python - {name}[*]')
        self.setWindowModified(dirty)

    def confirmDiscard(self):
        '''Ask before discarding unsaved changes.'''

        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self,
            'Unsaved Changes',
            'Save changes before continuing?',
            Buttons.Save | Buttons.Discard | Buttons.Cancel,
            Buttons.Save,
        )
        if answer == Buttons.Cancel:
            return False
        if answer == Buttons.Save:
            return self.saveDeck()
        return True

    def newDeck(self):
        '''Start a minimal runnable wire deck.'''

        if self.cards and not self.confirmDiscard():
            return
        self.cards = parseDeck(
            'CE New NEC-81 model\n'
            'GW 1,11,0,0,-0.25,0,0,0.25,0.001\n'
            'GE\n'
            'EX 0,1,6,0,1,0\n'
            'FR 0,1,0,0,299.792458,0\n'
            'RP 0,19,37,1000,0,0,10,10,0,0\n'
            'XQ\n'
            'EN\n'
        )
        self.deckPath = None
        self.outputPath.setText('model.out')
        self.plotDirectory.setText('plots')
        self.refreshAll()
        self.setDirty(False)

    def openDeck(self):
        """Prompt for and open a NEC deck or configuration file."""

        if not self.confirmDiscard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open NEC Input Deck or Config",
            "",
            "NEC decks & configs (*.nec *.cfg);;All files (*)",
        )
        if path:
            self.openDeckPath(path)

    def openDeckPath(self, path):
        """Open a known NEC deck or configuration path."""

        if looksLikeConfig(path):
            self.openConfigPath(path)
            return
        try:
            self.cards = loadDeck(path)
        except OSError as error:
            QMessageBox.critical(self, "Open Failed", str(error))
            return
        self.deckPath = Path(path).resolve()
        self.outputPath.setText(str(self.deckPath.with_suffix(".out")))
        self.plotDirectory.setText(str(self.deckPath.parent / "plots"))
        self.refreshAll()
        self.setDirty(False)
        self.setStatus(f"Opened {self.deckPath}")


    def openConfig(self):
        '''Prompt for a config file and load its deck and run settings.'''
        if not self.confirmDiscard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open NEC-81 Config',
            '',
            'Config files (*.cfg);;All files (*)',
        )
        if path:
            self.openConfigPath(path)

    def openConfigPath(self, path):
        '''Load GUI run settings and the input deck from a known config path.'''
        try:
            config = loadConfig(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, 'Config Error', str(error))
            return

        cfgPath = Path(path).resolve()
        if config.inputFile:
            self.openDeckPath(config.inputFile)
        elif config.embeddedDeck:
            self.cards = parseDeck(config.embeddedDeck)
            self.deckPath = None
            self.outputPath.setText(str(cfgPath.with_suffix('.out')))
            self.plotDirectory.setText(str(cfgPath.parent / 'plots'))
            self.refreshAll()
            self.setDirty(False)

        if config.outputFile:
            self.outputPath.setText(config.outputFile)
        if config.plotDirectory:
            self.plotDirectory.setText(config.plotDirectory)
        if config.plotFormat:
            self.plotFormat.setCurrentText(config.plotFormat)

        self.plots3d.setChecked(config.plots3d)
        self.plotGeometry.setChecked(config.plotGeometry)
        self.plotPatterns.setChecked(config.plotPatterns)
        self.plotFields.setChecked(config.plotFields)
        self.showPlots.setChecked(config.showPlots)
        self.showSources.setChecked(config.showSources)
        self.showLoads.setChecked(config.showLoads)
        self.writeAdf.setChecked(bool(config.writeAdf or config.adfFile))

        if config.embeddedDeck:
            self.setStatus(f'Opened embedded model from {cfgPath.name}; ready to run')
        else:
            self.setStatus(f'Opened config {cfgPath}')


    def saveDeck(self):
        '''Save to the current path, or ask for one.'''

        if self.deckPath is None:
            return self.saveDeckAs()
        try:
            saveDeck(self.deckPath, self.cards)
        except OSError as error:
            QMessageBox.critical(self, 'Save Failed', str(error))
            return False
        self.updateRawText()
        self.setDirty(False)
        self.setStatus(f'Saved {self.deckPath}')
        return True

    def saveDeckAs(self):
        '''Prompt for a new deck path and save it.'''

        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save NEC Input Deck',
            '',
            'NEC input decks (*.nec);;All files (*)',
        )
        if not path:
            return False
        if not Path(path).suffix:
            path += '.nec'
        self.deckPath = Path(path).resolve()
        if not self.outputPath.text().strip():
            self.outputPath.setText(str(self.deckPath.with_suffix('.out')))
        return self.saveDeck()

    def refreshAll(self):
        '''Synchronize all views from the card model.'''

        self.refreshCardTable()
        self.updateRawText()
        self.refreshGeometry()

    def refreshCardTable(self, selectedIndex=None):
        '''Rebuild the card table while retaining selection.'''

        if selectedIndex is None:
            selectedIndex = self.selectedCardIndex()
        self.cardTable.blockSignals(True)
        self.cardTable.setRowCount(len(self.cards))
        for index, card in enumerate(self.cards):
            number = QTableWidgetItem(str(index + 1))
            number.setTextAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )
            code = QTableWidgetItem(card.code)
            code.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cardTable.setItem(index, 0, number)
            self.cardTable.setItem(index, 1, code)
            self.cardTable.setItem(
                index, 2, QTableWidgetItem(cardSummary(card))
            )
        if self.cards:
            selectedIndex = max(0, min(selectedIndex, len(self.cards) - 1))
            self.cardTable.selectRow(selectedIndex)
            self.cardTable.setCurrentCell(selectedIndex, 0)
            self.cardTable.scrollToItem(self.cardTable.item(selectedIndex, 0))
        self.cardTable.blockSignals(False)
        if self.cards:
            self.loadCardEditor(selectedIndex)

    def selectedCardIndex(self):
        '''Return the selected card index or zero.'''

        return max(0, self.cardTable.currentRow())

    def selectCard(self):
        '''Load the selected card into the structured editor.'''

        row = self.cardTable.currentRow()
        if row >= 0:
            self.loadCardEditor(row)

    def loadCardEditor(self, index):
        '''Display one card's values in the structured editor.'''

        if not 0 <= index < len(self.cards):
            return
        card = self.cards[index]
        self.codeCombo.setCurrentText(card.code or 'GW')
        self.rebuildFields(card.values)

    def cardCodeText(self):
        '''Return the two-letter card type typed or chosen in the editor.'''

        return self.codeCombo.currentText().upper().strip()[:2]

    def changeCardType(self):
        '''Rebuild input fields when a card type changes.'''

        if self.cardCodeText() == self.shownCode:
            return
        values = [entry.text() for entry in self.fieldEntries]
        self.rebuildFields(values)

    def rebuildFields(self, values=None):
        '''Build labeled value entries for the selected card type.'''

        while self.fieldsForm.rowCount():
            self.fieldsForm.removeRow(0)
        self.fieldEntries = []
        code = self.cardCodeText()
        self.shownCode = code
        spec = CARD_SPECS.get(code)
        if spec:
            fields = spec.fields
            self.cardDescription.setText(f'{spec.name}: {spec.description}')
        else:
            fields = ('CommaSeparatedValues',)
            self.cardDescription.setText(
                'Unknown cards are preserved and may be edited as values.'
            )
        values = list(values or [])
        if spec is None and len(values) > 1:
            values = [','.join(values)]
        for row, fieldName in enumerate(fields):
            entry = QLineEdit()
            if row < len(values):
                entry.setText(values[row])
            self.fieldsForm.addRow(fieldName, entry)
            self.fieldEntries.append(entry)

    def cardFromEditor(self):
        '''Create a card from current structured editor values.'''

        code = self.cardCodeText()
        if len(code) != 2 or not code.isalpha():
            raise ValueError('Card type must contain exactly two letters.')
        values = [entry.text().strip() for entry in self.fieldEntries]
        if code not in CARD_SPECS and values:
            values = [value.strip() for value in values[0].split(',')]
        while values and values[-1] == '':
            values.pop()
        return DeckCard(code, values)

    def applyCardEdits(self):
        '''Replace the selected card with editor contents.'''

        if not self.cards:
            return
        try:
            card = self.cardFromEditor()
        except ValueError as error:
            QMessageBox.critical(self, 'Card Error', str(error))
            return
        index = self.selectedCardIndex()
        self.cards[index] = card
        self.afterCardChange(index)

    def insertCard(self):
        '''Insert editor contents after the selected card.'''

        try:
            card = self.cardFromEditor()
        except ValueError as error:
            QMessageBox.critical(self, 'Card Error', str(error))
            return
        index = self.selectedCardIndex() + 1 if self.cards else 0
        self.cards.insert(index, card)
        self.afterCardChange(index)

    def deleteCard(self):
        '''Delete the selected card.'''

        if not self.cards:
            return
        index = self.selectedCardIndex()
        del self.cards[index]
        self.afterCardChange(max(0, index - 1))

    def moveCard(self, offset):
        '''Move the selected card one row up or down.'''

        if not self.cards:
            return
        index = self.selectedCardIndex()
        destination = index + offset
        if not 0 <= destination < len(self.cards):
            return
        self.cards[index], self.cards[destination] = (
            self.cards[destination],
            self.cards[index],
        )
        self.afterCardChange(destination)

    def afterCardChange(self, selectedIndex):
        '''Refresh dependent views after a structured edit.'''

        self.refreshCardTable(selectedIndex)
        self.updateRawText()
        self.refreshGeometry()
        self.setDirty(True)

    def updateRawText(self):
        '''Replace raw editor contents from the card model.'''

        self.rawText.setPlainText(formatDeck(self.cards))

    def applyRawText(self):
        '''Parse raw editor contents into the card model.'''

        newCards = parseDeck(self.rawText.toPlainText())
        while newCards and not newCards[-1].code:
            newCards.pop()
        changed = formatDeck(newCards) != formatDeck(self.cards)
        self.cards = newCards
        self.refreshCardTable()
        self.refreshGeometry()
        if changed:
            self.setDirty(True)
        self.setStatus('Applied raw deck text')

    def validateCurrentDeck(self):
        '''Display structural deck validation messages.'''

        messages = validateDeck(self.cards)
        title = 'Deck Validation'
        body = '\n'.join(messages)
        if any(message.startswith('Error') for message in messages):
            QMessageBox.warning(self, title, body)
        else:
            QMessageBox.information(self, title, body)
        self.setStatus(messages[0])

    def refreshGeometry(self):
        '''Redraw the wire and surface-patch geometry preview.'''

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        axis = self.geometryAxis
        axis.clear()
        try:
            model = previewModel(self.cards)
            segments = [(wire[1], wire[2]) for wire in model.wires]
            patches = model.patches
        except ValueError as error:
            axis.text2D(0.05, 0.95, str(error), transform=axis.transAxes)
            self.geometryCanvas.draw_idle()
            return
        if patches:
            axis.add_collection3d(Poly3DCollection(
                patches,
                facecolors='tab:orange',
                edgecolors='saddlebrown',
                linewidths=0.6,
                alpha=0.35,
            ))
        for start, end in segments:
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color='tab:blue',
                linewidth=1.6,
            )
        center, radius_m = geometryBounds(segments, patches)
        axis.set_xlim(center[0] - radius_m, center[0] + radius_m)
        axis.set_ylim(center[1] - radius_m, center[1] + radius_m)
        axis.set_zlim(center[2] - radius_m, center[2] + radius_m)
        axis.set_xlabel('x (m)')
        axis.set_ylabel('y (m)')
        axis.set_zlabel('z (m)')
        axis.set_title(
            f'Geometry ({len(segments)} segments, {len(patches)} patches)'
        )
        self.geometryFigure.tight_layout()
        self.geometryCanvas.draw_idle()

    def chooseOutputPath(self):
        '''Choose the text report written by the solver.'''

        initialFile = Path(self.outputPath.text() or 'model.out')
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Choose NEC-81 Report File',
            str(initialFile),
            'NEC reports (*.out);;All files (*)',
        )
        if path:
            self.outputPath.setText(path)


    def choosePlotDirectory(self):
        '''Choose the directory used for saved plot files.'''

        path = QFileDialog.getExistingDirectory(
            self,
            'Choose Plot Directory',
            self.plotDirectory.text() or '.',
        )
        if path:
            self.plotDirectory.setText(path)


    def makeRunConfig(self):
        '''Write current GUI settings to a temporary solver config.'''

        if self.deckPath is None:
            deckFile = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.nec',
                prefix='nec81_gui_',
                delete=False,
                encoding='utf-8',
            )
            with deckFile:
                deckFile.write(formatDeck(self.cards))
            self.temporaryDeck = Path(deckFile.name)
            runDeckPath = self.temporaryDeck
            baseDirectory = Path.cwd()
        else:
            runDeckPath = self.deckPath
            baseDirectory = self.deckPath.parent
        outputPath = Path(self.outputPath.text().strip() or 'model.out')
        plotDirectory = Path(self.plotDirectory.text().strip() or 'plots')
        if not outputPath.is_absolute():
            outputPath = baseDirectory/outputPath
        if not plotDirectory.is_absolute():
            plotDirectory = baseDirectory/plotDirectory
        outputPath = outputPath.resolve()

        plotDirectory = plotDirectory.resolve()
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        self.outputPath.setText(str(outputPath))
        self.plotDirectory.setText(str(plotDirectory))

        def boolText(checkBox):
            return 'true' if checkBox.isChecked() else 'false'

        plotFormat = self.plotFormat.currentText().strip() or 'png'
        lines = (
            f'inputFile "{runDeckPath}"',
            f'outputFile "{outputPath}"',
            f'plots3d {boolText(self.plots3d)}',
            f'plotGeometry {boolText(self.plotGeometry)}',
            f'plotPatterns {boolText(self.plotPatterns)}',
            f'plotFields {boolText(self.plotFields)}',
            f'plotDirectory "{plotDirectory}"',
            f'plotFormat {plotFormat}',
            f'showPlots {boolText(self.showPlots)}',
            f'showSources {boolText(self.showSources)}',
            f'showLoads {boolText(self.showLoads)}',
            f'writeAdf {boolText(self.writeAdf)}'
        )
        configFile = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.cfg',
            prefix='nec81_gui_',
            delete=False,
            encoding='utf-8',
        )
        with configFile:
            configFile.write('\n'.join(lines) + '\n')
        self.temporaryConfig = Path(configFile.name)
        return self.temporaryConfig

    def removeTemporaryFiles(self):
        '''Delete the temporary config and deck written for a run.'''

        for name in ('temporaryConfig', 'temporaryDeck'):
            path = getattr(self, name)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                setattr(self, name, None)

    def runSolver(self):
        '''Save the deck and run NEC-81 in a background process.'''

        if self.process is not None:
            QMessageBox.information(
                self, 'Run Active', 'NEC-81 is already running.'
            )
            return
        self.stopRequested = False
        self.applyRawText()
        errors = [
            message for message in validateDeck(self.cards)
            if message.startswith('Error')
        ]
        if errors:
            QMessageBox.critical(self, 'Deck Validation', '\n'.join(errors))
            return
        if self.deckPath is not None and not self.saveDeck():
            return
        try:
            configPath = self.makeRunConfig()
        except (OSError, ValueError) as error:
            self.finishRun(None, str(error))
            return
        srcDirectory = str(Path(__file__).resolve().parents[1])
        environment = QProcessEnvironment.systemEnvironment()
        pythonPath = srcDirectory
        oldPythonPath = os.environ.get('PYTHONPATH', '')
        if oldPythonPath:
            pythonPath += os.pathsep + oldPythonPath
        environment.insert('PYTHONPATH', pythonPath)

        arguments = [
            '-u', '-m', 'nec81.nec81', '--config', str(configPath)
        ]
        self.clearProcessText()
        self.appendProcessText(
            'Running: ' + ' '.join([sys.executable] + arguments) + '\n'
        )
        self.outputDecoder = codecs.getincrementaldecoder('utf-8')(
            errors='replace'
        )

        self.process = QProcess(self)
        self.process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels
        )
        self.process.setProcessEnvironment(environment)
        runDirectory = (
            str(self.deckPath.parent)
            if self.deckPath is not None
            else str(Path.cwd())
        )
        self.process.setWorkingDirectory(runDirectory)
        self.process.readyReadStandardOutput.connect(self.readProcessOutput)
        self.process.finished.connect(self.onProcessFinished)
        self.process.errorOccurred.connect(self.onProcessError)
        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.setStatus('NEC-81 is running')
        self.notebook.setCurrentWidget(self.runTab)
        self.process.start(sys.executable, arguments)

    def readProcessOutput(self):
        '''Append any new child-process output to the output pane.'''

        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput())
        self.appendProcessText(self.outputDecoder.decode(data))

    def onProcessFinished(self, returnCode, exitStatus):
        '''Collect final output when the child process exits.'''

        self.readProcessOutput()
        self.appendProcessText(self.outputDecoder.decode(b'', final=True))
        self.finishRun(returnCode)

    def onProcessError(self, error):
        '''Report a process that could not be started.'''

        if error == QProcess.ProcessError.FailedToStart:
            message = self.process.errorString()
            self.finishRun(None, message)

    def finishRun(self, returnCode, errorMessage=None):
        '''Restore controls and load products after the process exits.'''

        if self.process is not None:
            self.process.deleteLater()
            self.process = None
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.removeTemporaryFiles()
        if errorMessage:
            self.appendProcessText(f'\nGUI runner error: {errorMessage}\n')
            self.setStatus('Run could not be started')
            QMessageBox.critical(self, 'Run Failed', errorMessage)
            return
        self.appendProcessText(f'\nNEC-81 exited with status {returnCode}.\n')
        if self.stopRequested:
            self.setStatus('Run stopped')
        elif returnCode == 0:
            self.setStatus('Run completed')
            self.loadReport(selectTab=False)
            self.loadPlotFiles()
            if Path(self.outputPath.text()).is_file():
                self.notebook.setCurrentWidget(self.reportTab)
        else:
            self.setStatus(f'Run failed with status {returnCode}')
            QMessageBox.critical(
                self,
                'Run Failed',
                f'NEC-81 exited with status {returnCode}.',
            )

    def stopSolver(self):
        '''Request termination of the active NEC-81 process.'''

        if self.process is None:
            return
        self.stopRequested = True
        self.appendProcessText('\nTermination requested.\n')
        self.setStatus('Stopping NEC-81')
        self.stopButton.setEnabled(False)
        if sys.platform == 'win32':
            # terminate() cannot stop a console process on Windows.
            self.process.kill()
        else:
            self.process.terminate()
            QTimer.singleShot(3000, self.killIfRunning)

    def killIfRunning(self):
        '''Force the process to end if it ignored the terminate request.'''

        if (self.process is not None
                and self.process.state() != QProcess.ProcessState.NotRunning):
            self.process.kill()

    def clearProcessText(self):
        '''Clear the process-output pane.'''

        self.processText.clear()

    def appendProcessText(self, text):
        '''Append text to the read-only process-output pane.'''

        if not text:
            return
        self.processText.moveCursor(QTextCursor.MoveOperation.End)
        self.processText.insertPlainText(text)
        self.processText.ensureCursorVisible()

    def loadReport(self, selectTab=True):
        '''Load the configured solver report into its viewer.'''

        reportPath = Path(self.outputPath.text().strip() or 'model.out')
        try:
            text = reportPath.read_text(encoding='utf-8', errors='replace')
        except OSError as error:
            self.reportText.setPlainText(
                f'Report is not available:\n{error}\n'
            )
            return
        self.reportText.setPlainText(text)

        summary_status = f'Loaded report {reportPath}'
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if 'ANTENNA INPUT PARAMETERS' in line:
                for sub in lines[idx + 1 : idx + 12]:
                    parts = sub.split()
                    if len(parts) >= 8 and parts[0].isdigit() and parts[1].isdigit():
                        try:
                            r_ohms = float(parts[6])
                            x_ohms = float(parts[7])
                            z_c = complex(r_ohms, x_ohms)
                            gamma = abs((z_c - 50.0) / (z_c + 50.0))
                            vswr = min((1.0 + gamma) / max(1.0 - gamma, 1e-6), 99.9)
                            sign_x = '+' if x_ohms >= 0.0 else '-'
                            summary_status = (
                                f'Done — Z = {r_ohms:.2f} {sign_x} j{abs(x_ohms):.2f} Ω  |  '
                                f'VSWR (50 Ω) = {vswr:.2f}:1'
                            )
                        except ValueError:
                            pass
                        break
                break

        self.setStatus(summary_status)
        if selectTab:
            self.notebook.setCurrentWidget(self.reportTab)

    def loadPlotFiles(self):
        '''Find saved solver plots and update the plot chooser.'''

        directory = Path(self.plotDirectory.text().strip() or 'plots')
        extensions = {'.bmp', '.jpeg', '.jpg', '.pdf', '.png', '.svg',
                      '.tif', '.tiff'}
        try:
            self.plotFiles = sorted(
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            )
        except OSError:
            self.plotFiles = []
        names = [path.name for path in self.plotFiles]
        current = self.plotCombo.currentText()
        self.plotCombo.clear()
        self.plotCombo.addItems(names)
        if names:
            self.plotCombo.setCurrentText(
                current if current in names else names[0]
            )
            self.displayPlot()
        else:
            self.plotAxis.clear()
            self.plotAxis.axis('off')
            self.plotAxis.text(
                0.5,
                0.5,
                f'No saved plots found in\n{directory}',
                ha='center',
                va='center',
                transform=self.plotAxis.transAxes,
            )
            self.plotCanvas.draw_idle()

    def displayPlot(self):
        '''Display the selected raster plot in the embedded viewer.'''

        selected = self.plotCombo.currentText()
        plotPath = next(
            (path for path in self.plotFiles if path.name == selected),
            None,
        )
        if plotPath is None:
            return
        self.plotAxis.clear()
        self.plotAxis.axis('off')
        if plotPath.suffix.lower() in {'.pdf', '.svg'}:
            self.plotAxis.text(
                0.5,
                0.5,
                f'{plotPath.name}\n\n'
                'Vector plots are saved but not embedded.\n'
                'Use PNG for the in-application preview.',
                ha='center',
                va='center',
                transform=self.plotAxis.transAxes,
            )
        else:
            try:
                from matplotlib.image import imread

                image = imread(plotPath)
                self.plotAxis.imshow(image)
                self.plotAxis.set_title(plotPath.name)
            except (OSError, ValueError) as error:
                self.plotAxis.text(
                    0.5,
                    0.5,
                    f'Could not load {plotPath.name}:\n{error}',
                    ha='center',
                    va='center',
                    transform=self.plotAxis.transAxes,
                )
        self.plotFigure.tight_layout()
        self.plotCanvas.draw_idle()

    def showHelp(self):
        '''Show a concise summary of the GUI workflow.'''

        QMessageBox.information(
            self,
            'NEC-81 GUI Help',
            'Open or create a NEC deck, edit common cards in Card Editor, '
            'and use Raw Deck for advanced cards.\n\n'
            'Apply raw changes before returning to structured editing. '
            'The Geometry tab previews GW wires.\n\n'
            'Choose report and plot locations on the Run tab, then select '
            'Run NEC-81 (F5). Interactive plot windows remain open when '
            'that option is enabled.\n\n'
            'See pythonNec81/GUI.md for complete instructions.',
        )

    def showAbout(self):
        '''Display application information.'''

        QMessageBox.about(
            self,
            'About NEC-81 Python',
            'NEC-81 Python desktop interface\n\n'
            'Edits ordinary NEC input decks and runs the Python port '
            'through its existing configuration and command-line backend.',
        )

    def closeEvent(self, event):
        '''Confirm pending work, stop a run, and close the application.'''

        if not self.confirmDiscard():
            event.ignore()
            return
        if self.process is not None:
            answer = QMessageBox.question(
                self,
                'Run Active',
                'Stop the active NEC-81 run and exit?',
            )
            if answer != Buttons.Yes:
                event.ignore()
                return
            self.stopRequested = True
            self.process.kill()
            self.process.waitForFinished(2000)
        self.removeTemporaryFiles()
        event.accept()


def launchGui(initialDeck=None):
    '''Create and run the optional Qt desktop application.'''

    application = QApplication.instance() or QApplication(sys.argv[:1])
    window = Nec81Gui(initialDeck=initialDeck)
    window.show()
    application.exec()


def main(argv=None):
    '''Launch the GUI directly, optionally opening one deck.'''

    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) > 1:
        raise SystemExit('usage: python -m pythonNec81.gui [DECK]')
    launchGui(arguments[0] if arguments else None)


if __name__ == '__main__':
    main()
