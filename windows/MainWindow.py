import datetime

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QScrollArea,
    QTextBrowser,
    QLabel,
    QSlider,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QFileDialog,
)

from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt

import os

import pandas as pd

from constants.paths import RESULT_PATH
from widgets.ImageBrowser import ImageBrowser

from pathlib import WindowsPath

from torch import cuda
import torch

from threading import Thread

from collections import defaultdict

from dino.dino import classify
from transformers import Dinov2ForImageClassification

import windows.database as database
from sqlalchemy import select

from PIL import Image

import webbrowser


def loadPaths(startpath: str, tree : QTreeWidget) -> None:
    '''
    Рекурсивная подгрузка файлов в древо файлов
    '''
    for elem in os.listdir(startpath):
        elemPath = startpath + "/" + elem
        parentalItem = QTreeWidgetItem(tree, [os.path.basename(elem)])

        if os.path.isdir(elemPath):
            loadPaths(elemPath, parentalItem)
            parentalItem.setIcon(0, QIcon("assets\open-folder.png"))
        else:
            parentalItem.setIcon(0, QIcon("assets\processing.png"))


def getItemFullPath(item : QTreeWidgetItem) -> str:
    '''
    Рекурсивное получение полного пути члена дерева файлов
    '''
    out = item.text(0)

    if item.parent():
        out = getItemFullPath(item.parent()) + "/" + out
    else:
        out =  "results/" + out

    return out

NUM_CLASSES = 3

class MainWindow(QMainWindow):
    currentlySelectedFolder = ""
    labelsNotShowed = False

    def __init__(self) -> None:
        super().__init__()
        self.model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base")
        self.model.classifier = torch.nn.Linear(768, NUM_CLASSES)
        # self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg_lc", pretrained=True)
        # self.model.linear_head = torch.nn.Linear(768, 3)
        

        self.initUi()


    def initUi(self):
        '''
        Инициализация UI
        '''
        self.setWindowTitle("олени или кто?")
        self.setWindowIcon(QIcon("assets/deer.png"))


        # Получение размеров экрана ===========================================
        screen = QApplication.primaryScreen()
        screenSize = screen.size()

        self.windowWidth  = screenSize.width()
        self.windowHeight = screenSize.height()
        # ======================================================================


        # Создание лейаутов приложения =========================================
        self.primaryLayout   = QHBoxLayout()
        self.firstColLayout  = QVBoxLayout()
        self.secondColLayout = QVBoxLayout()
        self.thirdColLayout  = QVBoxLayout()
        # ======================================================================


        # Вызов функций создания основных блоков GUI ===========================
        self.initResultsAndControls()
        self.initImageBrowser()
        # self.initGraphsAndInfo()
        # ======================================================================


        # Установка ориентаций элементов в лейаутах ============================
        self.secondColLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.firstColLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # ======================================================================


        # Добавление вторичных лейаутов в основной =============================
        self.primaryLayout.addLayout(self.firstColLayout)
        self.primaryLayout.addLayout(self.secondColLayout)
        self.primaryLayout.addLayout(self.thirdColLayout)
        # ======================================================================


        # Виджет-контейнер для остальных элментов GUI ==========================
        self.container = QWidget()
        self.container.setLayout(self.primaryLayout)
        # ======================================================================

        self.setCentralWidget(self.container)


    def initResultsAndControls(self):
        '''
        Инициализация древа файлов и папок в результатах и контроля параметров модели
        '''

        # Виджет древа файлов в папке результатов
        self.resultFileTree = QTreeWidget()
        self.resultFileTree.setHeaderLabel("Results")
        self.resultFileTree.setMaximumWidth(int(self.windowWidth / 5))
        self.resultFileTree.setMinimumWidth(int(self.windowWidth / 6))
        self.resultFileTree.setMaximumHeight(int(self.windowHeight / 2))
        self.resultFileTree.itemClicked.connect(self.treeItemClicked)

        loadPaths(startpath=RESULT_PATH, tree=self.resultFileTree)

        # Чекбоксы девайсов для работы модели ==================================
        self.deviceCheckBoxesLabel = QLabel("Девайс")
        self.deviceCheckBoxesLabel.setMaximumHeight(15)
        self.deviceCheckBoxesLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.deviceCheckBoxesLayout = QHBoxLayout()

        self.cpuCheckBox  = QCheckBox("CPU")
        self.cpuCheckBox.setMaximumWidth(50)
        self.cpuCheckBox.setChecked(True)
        self.cpuCheckBox.stateChanged.connect(self.checkBoxCheckMate)

        self.cudaCheckBox = QCheckBox("GPU") 
        self.cudaCheckBox.setEnabled(cuda.is_available())
        self.cudaCheckBox.stateChanged.connect(self.checkBoxCheckMate)
        self.cudaCheckBox.setMaximumWidth(50)

        self.deviceCheckBoxesLayout.addWidget(self.cpuCheckBox)
        self.deviceCheckBoxesLayout.addWidget(self.cudaCheckBox)
        # ======================================================================


        # self.arelabelsNotShowedCheckBox = QCheckBox("Режим классификации (без детекции)")
        # self.arelabelsNotShowedCheckBox.stateChanged.connect(self.labelsCheckBoxChange)
        # self.arelabelsNotShowedCheckBox.setMaximumWidth(140)


        # Окно с отображением пути выбранной папки =============================
        self.pathLayout = QHBoxLayout()

        self.pathLabel = QLabel("Выбранная папка:")
        self.pathLabel.setMaximumHeight(15)
        self.pathLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.pathLabel.setMaximumWidth(100)

        self.pathDisplay = QLineEdit(self.currentlySelectedFolder)
        self.pathDisplay.setMaximumWidth(int(self.windowWidth / 5) - 100)
        self.pathDisplay.setReadOnly(True)

        self.pathLayout.addWidget(self.pathLabel)
        self.pathLayout.addWidget(self.pathDisplay)
        # ======================================================================


        # Кнопка выбора папки ==================================================
        self.chooseFolderButton = QPushButton("Выбрать папку")
        self.chooseFolderButton.clicked.connect(self.chooseFolder)
        # ======================================================================


        # Кнопка детект ========================================================
        # self.detectButton = QPushButton("Detect")
        # self.detectButton.clicked.connect(self.onDetectButtonClicked)
        # self.detectButton.setEnabled(False)
        # ======================================================================

        # Кнопка классификации
        self.classifyButton = QPushButton("Классифицировать")
        self.classifyButton.clicked.connect(self.onClassifyButtonClicked)
        self.classifyButton.setEnabled(False)

        # Кнопка открытия дашборда
        self.openDashboardButton = QPushButton("Открыть статистику")
        self.openDashboardButton.clicked.connect(self.onOpenDashboardClicked)
        self.openDashboardButton.setEnabled(True)

        # Лейбл с результатами последней классификации
        self.lastClassificationLabel = QLabel("")
        self.lastClassificationLabel.setFont(QFont("Arial", 14))

        # Добавление виджетов и лейаутов по их колонкам ========================
        self.firstColLayout.addWidget(self.resultFileTree)
        self.firstColLayout.addWidget(self.deviceCheckBoxesLabel)
        self.firstColLayout.addLayout(self.deviceCheckBoxesLayout)
        self.firstColLayout.addLayout(self.pathLayout)
        self.firstColLayout.addWidget(self.chooseFolderButton)
        self.firstColLayout.addWidget(self.classifyButton)
        self.firstColLayout.addWidget(self.openDashboardButton)
        self.firstColLayout.addWidget(self.lastClassificationLabel, alignment=Qt.AlignmentFlag.AlignTop)
        # ======================================================================


    def checkBoxCheckMate(self):
        '''
        Функция проверки чекбоксов и соответствующей смены их состояний
        '''
        if self.sender().checkState() == Qt.CheckState.Checked:
            if   self.sender() == self.cpuCheckBox:
                self.cudaCheckBox.setChecked(False)
            elif self.sender() == self.cudaCheckBox:
                self.cpuCheckBox.setChecked(False)


    def chooseFolder(self):
        '''
        Функция вызываемая при нажатии на кнопку выбора папки с изображениями
        '''
        currentUser = os.environ.get('USER', os.environ.get('USERNAME'))
        self.currentlySelectedFolder = QFileDialog(directory=f"C:/Users/{currentUser}/Pictures").getExistingDirectory()

        self.updatePathDisplay()


    def updatePathDisplay(self):
        '''
        Функция обновленя отображения окна пути выбранной в данный момент папки
        '''
        self.pathDisplay.setText(self.currentlySelectedFolder)
        if self.currentlySelectedFolder != "":
            # self.detectButton.setEnabled(True)
            self.classifyButton.setEnabled(True)
        else:
            # self.detectButton.setEnabled(False)
            self.classifyButton.setEnabled(False)


    def initImageBrowser(self):
        '''
        Инициализация браузера изображений
        '''
        self.imageBrowser = ImageBrowser()
        self.imageBrowser.updatePixmap(None)
        self.imageBrowser.setMaximumWidth(int(self.windowWidth / 2))

        self.imageTitle = QLabel("")
        self.imageTitle.setMaximumHeight(60)
        self.imageTitle.setFont(QFont("Arial", 30))

        self.secondColLayout.addWidget(self.imageTitle, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.secondColLayout.addWidget(self.imageBrowser, alignment=Qt.AlignmentFlag.AlignHCenter)


    def onClassifyButtonClicked(self):
        classify_thread = Thread(name="classifier_thread", target=self.classify)
        classify_thread.start()
        
        self.classifyButton.setEnabled(False)

    def classify(self):
        if self.cudaCheckBox.isChecked(): device = "cuda:0"
        else: device = "cpu"

        imgs = list(map(lambda path: str(self.currentlySelectedFolder + "/" + path), os.listdir(self.currentlySelectedFolder)))
        print(imgs)
        classes = classify(imgs, self.model, device)
        self.saveImgsToResults(imgs)
        self.resultFileTree.clear()
        loadPaths(RESULT_PATH, self.resultFileTree)

        self.lastClassificationLabel.setText(
            f"Данные последней классификации:\n- Кабарга: {classes.count('Кабарга')}\n- Олень: {classes.count('Олень')}\n- Косуля: {classes.count('Косуля')}"
        )


    def saveImgsToResults(self, imgsPaths : list[str]):
        runName = f"run_{self.get_current_datetime()}"
        os.mkdir(os.path.join(RESULT_PATH, runName))
        for imgPath in imgsPaths:
            img = Image.open(imgPath)
            imgName = os.path.basename(imgPath)
            img.save(os.path.join(RESULT_PATH, runName, imgName))


    def get_current_datetime(self):
        return datetime.datetime.now().strftime("%I-%M%p_%B_%d_%Y")
    
    
    def onOpenDashboardClicked(self):
        import dashboard.temp as dashboard
        dashboardThread = Thread(name="dashboard_thread", target=dashboard.app.run_server)
        dashboardThread.start()
        webbrowser.open("http://127.0.0.1:8050/")
        

    def treeItemClicked(self, it : QTreeWidgetItem, col):
        '''
        Функция вызываемая при нажатии на элемент в дереве файлов папки результатов
        '''
        if not os.path.isdir(getItemFullPath(it)):
            self.imageBrowser.updatePixmap(getItemFullPath(it))
            path = getItemFullPath(it)
            imgName = os.path.basename(path)
            print(imgName)
            if ((".png" in imgName) or (".jpg" in imgName) or (".JPEG" in imgName)):
                result = database.session.scalars(select(database.Image).where(database.Image.name == imgName))
                imgClass = result.all()[0]
                self.imageTitle.setText(imgClass.class_)