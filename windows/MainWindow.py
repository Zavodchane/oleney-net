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
    QFileDialog
)

from PyQt6.QtGui import QIcon
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


def accuracy(detectOutput : dict, filesCount) -> float:
    classNameToClassCode = {
        "shipun" : 3,
        "klikun" : 2,
        "small"  : 1
    }

    data = defaultdict(list)

    correctFiles = 0
    labelsNotShowed = False

    for fileName, labelAccTupleList in detectOutput.items():

        maxConfTuple = max(labelAccTupleList, key = lambda labelAccTuple: labelAccTuple[1])
        
        className = maxConfTuple[0]

        print(className, fileName)

        if ("img" or "shipun") in fileName:
            if className == "shipun":
                correctFiles += 1
        elif (className in fileName) and (className != ""):
            correctFiles += 1

        data["name"].append(fileName)
        data["class"].append(classNameToClassCode[className])

    dataFrame = pd.DataFrame(data)
    dataFrame.to_csv(sep=";", encoding="utf-8", path_or_buf="tables\\output.csv", index = False)

    return (correctFiles / filesCount) * 100

NUM_CLASSES = 3

class MainWindow(QMainWindow):
    currentlySelectedFolder = ""
    labelsNotShowed = False

    def __init__(self) -> None:
        super().__init__()
        self.model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base")
        self.model.classifier = torch.nn.Linear(768, NUM_CLASSES)
        

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
        self.secondColLayout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
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

        # Слайдер процента уверенности ========================================
        # self.detectionPercentSliderLayout = QHBoxLayout()

        # self.detectionSliderLabel = QLabel("Минимальный процент уверенности")
        # self.detectionSliderLabel.setMaximumHeight(15)
        # self.detectionSliderLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # self.detectionPercentLabel = QLabel("50%")
        # self.detectionPercentLabel.setMaximumWidth(30)
        # self.detectionPercentLabel.setAlignment(Qt.AlignmentFlag.AlignRight)

        # self.detectionPercentSlider = QSlider(Qt.Orientation.Horizontal)
        # self.detectionPercentSlider.setMaximumWidth(int(self.windowWidth / 5) - 35)
        # self.detectionPercentSlider.setRange(1, 100)
        # self.detectionPercentSlider.setValue(50)
        # self.detectionPercentSlider.setSingleStep(1)
        # self.detectionPercentSlider.valueChanged.connect(self.onPercentSliderValueChange)

        # self.detectionPercentSliderLayout.addWidget(self.detectionPercentLabel)
        # self.detectionPercentSliderLayout.addWidget(self.detectionPercentSlider)
        # ======================================================================


        # Слайдер толщины линии обводки лебедей ================================
        # self.lineThicknessSliderLayout = QHBoxLayout()

        # self.lineThicknessSliderLabel = QLabel("Толщина линий выделения")
        # self.lineThicknessSliderLabel.setMaximumHeight(15)
        # self.lineThicknessSliderLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # self.lineThicknessLabel = QLabel("3")
        # self.lineThicknessLabel.setMaximumWidth(30)
        # self.lineThicknessLabel.setAlignment(Qt.AlignmentFlag.AlignRight)

        # self.lineThicknessSlider = QSlider(Qt.Orientation.Horizontal)
        # self.lineThicknessSlider.setMaximumWidth(int(self.windowWidth / 5) - 35)
        # self.lineThicknessSlider.setRange(0, 10)
        # self.lineThicknessSlider.setValue(3)
        # self.lineThicknessSlider.setSingleStep(1)
        # self.lineThicknessSlider.valueChanged.connect(self.onThicknessSliderValueChange)

        # self.lineThicknessSliderLayout.addWidget(self.lineThicknessLabel)
        # self.lineThicknessSliderLayout.addWidget(self.lineThicknessSlider)
        # ======================================================================


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


        # Добавление виджетов и лейаутов по их колонкам ========================
        self.firstColLayout.addWidget(self.resultFileTree)
        # self.firstColLayout.addWidget(self.detectionSliderLabel)
        # self.firstColLayout.addLayout(self.detectionPercentSliderLayout)
        # self.firstColLayout.addWidget(self.lineThicknessSliderLabel)
        # self.firstColLayout.addLayout(self.lineThicknessSliderLayout)
        self.firstColLayout.addWidget(self.deviceCheckBoxesLabel)
        self.firstColLayout.addLayout(self.deviceCheckBoxesLayout)
        # self.firstColLayout.addWidget(self.arelabelsNotShowedCheckBox)
        self.firstColLayout.addLayout(self.pathLayout)
        self.firstColLayout.addWidget(self.chooseFolderButton)
        # self.firstColLayout.addWidget(self.detectButton)
        self.firstColLayout.addWidget(self.classifyButton)
        # ======================================================================


    # def onPercentSliderValueChange(self):
    #     '''
    #     Функция вызываемая при смене значения на слайдере
    #     '''
    #     self.detectionPercentLabel.setText(str(self.sender().value()) + "%")

    
    # def onThicknessSliderValueChange(self):
    #     '''
    #     Функция вызываемая при смене значения на слайдере
    #     '''
    #     self.lineThicknessLabel.setText(str(self.sender().value()))


    def checkBoxCheckMate(self):
        '''
        Функция проверки чекбоксов и соответствующей смены их состояний
        '''
        if self.sender().checkState() == Qt.CheckState.Checked:
            if   self.sender() == self.cpuCheckBox:
                self.cudaCheckBox.setChecked(False)
            elif self.sender() == self.cudaCheckBox:
                self.cpuCheckBox.setChecked(False)


    # def labelsCheckBoxChange(self):
    #     if self.sender().checkState() == Qt.CheckState.Checked:
    #         self.labelsNotShowed = True
    #         self.lineThicknessSlider.setValue(0)
    #     else:
    #         self.labelsNotShowed = False
    #         self.lineThicknessSlider.setValue(3)

        
    #     self.lineThicknessSlider.setEnabled(not self.labelsNotShowed)


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

        self.secondColLayout.addWidget(self.imageBrowser, alignment=Qt.AlignmentFlag.AlignTop)


    # @staticmethod
    # def _get_swans_quantity(label: str) -> list:
    #     months_swan_quantity = [0 for i in range(12)]
    #     for swan_data in statistic_db.get_records_by_name(label):
    #         month = int(swan_data[2].split('.')[1])-1
    #         months_swan_quantity[month] += swan_data[1]
    #     return months_swan_quantity


    def onClassifyButtonClicked(self):
        classify_thread = Thread(target=self.classify)
        classify_thread.start()
        
        self.classifyButton.setEnabled(False)

    # def onDetectButtonClicked(self):
    #     '''
    #     Функция вызываемая при нажатии на кнопку Detect, создает отдельный поток для выполнения детектирования
    #     '''
    #     detection_thread = Thread(target=self.detect)
    #     detection_thread.start()

    def classify(self):
        if self.cudaCheckBox.isChecked(): device = "cuda:0"
        else: device = "cpu"

        imgs = list(map(lambda path: str(self.currentlySelectedFolder + "/" + path), os.listdir(self.currentlySelectedFolder)))
        print(imgs)
        print(classify(imgs, self.model, device))

    
    # def detect(self):
    #     '''
    #     Функция детектирования
    #     '''
    #     # self.detectButton.setEnabled(False)

    #     if self.cudaCheckBox.isChecked():
    #         device = "cuda:0"
    #     else: 
    #         device = "cpu"

    #     # Аргументы для модели =================================================
    #     options = {
    #         'weights': ['models\\best_yolov5x_50e.pt'], 
    #         'source': self.currentlySelectedFolder, 
    #         'data': WindowsPath('data/coco128.yaml'), 
    #         'imgsz': [640, 640], 
    #         'conf_thres': self.detectionPercentSlider.value()/100, 
    #         'iou_thres': 0.45, 
    #         'max_det': 1000, 
    #         'device': device, 
    #         'view_img': False, 
    #         'save_txt': False, 
    #         'save_conf': False, 
    #         'save_crop': False, 
    #         'nosave': False, 
    #         'classes': None, 
    #         'agnostic_nms': False, 
    #         'augment': False, 
    #         'visualize': False, 
    #         'update': False, 
    #         'project': WindowsPath('results'), 
    #         'name': 'run', 
    #         'exist_ok': False, 
    #         'line_thickness': 3,
    #         'hide_labels': self.labelsNotShowed, 
    #         'hide_conf': False, 
    #         'half': False, 
    #         'dnn': False, 
    #         'vid_stride': 1
    #     }
    #     # ======================================================================

        
    #     # Вызов функции детектирования =========================================
    #     # detectResults = detect(options)
    #     en_to_rus = {'small': 'Малые лебеди',
    #                  'klikun': 'Кликуны',
    #                  'shipun': 'Шипуны'}
    #     date = str(datetime.datetime.now()).split()[0]
    #     date = date.replace('-', '.')
    #     date = date.split('.')
    #     date = f"{date[2]}.{date[1].replace('0', '')}.{date[0]}"
    #     # for _, val in detectResults.items():
    #     #     for swan in val:
    #     #         statistic_db.add_record(swan_name=en_to_rus[swan[0]],
    #     #                                 quantity=1,
    #     #                                 date=date)

    #     # print(f"Точность при валидации: {accuracy(detectResults, len(os.listdir(self.currentlySelectedFolder)))}%")
    #     # ======================================================================

    #     # Обновление древа файлов результатов после детектирования =============
    #     self.resultFileTree.clear()
    #     loadPaths(RESULT_PATH, self.resultFileTree)
    #     # ======================================================================

    #     # Обновления поля выбранной в данный момент папки ======================
    #     self.currentlySelectedFolder = ""
    #     self.updatePathDisplay()
    #     # ======================================================================


    def treeItemClicked(self, it : QTreeWidgetItem, col):
        '''
        Функция вызываемая при нажатии на элемент в дереве файлов папки результатов
        '''
        if not os.path.isdir(getItemFullPath(it)):
            self.imageBrowser.updatePixmap(getItemFullPath(it))