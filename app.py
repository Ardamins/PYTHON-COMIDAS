# -*- coding: utf-8 -*-
import sys
import os
import shutil
import csv
import json
import random
import time
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QInputDialog, QTableWidget, QTableWidgetItem,
    QDialog, QProgressBar, QComboBox, QFrame, QScrollArea, QGridLayout,
    QSizePolicy, QGraphicsOpacityEffect, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen
from PyQt5.QtCore import (
    Qt, QStandardPaths, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QThread
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from fpdf import FPDF


# ------------------------------------------------------------------
# Importación de TFLite con triple fallback
# ------------------------------------------------------------------
tflite = None
_USE_TFLITE_RUNTIME = False

try:
    import tflite_runtime.interpreter as tflite
    _USE_TFLITE_RUNTIME = True
except ImportError:
    try:
        from tensorflow.lite.python import interpreter as tflite
    except ImportError:
        try:
            import tensorflow as tf
            tflite = tf.lite
        except ImportError:
            tflite = None


try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


def _abortar_sin_tflite():
    from PyQt5.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.critical(
        None, "Error de dependencias",
        "No se pudo cargar ningún intérprete TFLite.\n\n"
        "Instala una de estas opciones en tu entorno virtual:\n"
        "  pip install --index-url https://google-coral.github.io/py-repo/ tflite-runtime\n"
        "o bien:\n"
        "  pip install tensorflow-cpu==2.16.2"
    )
    sys.exit(1)


# ------------------------------------------------------------------
# Rutas adaptadas para PyInstaller
# ------------------------------------------------------------------
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def writable_path(relative_path=""):
    if getattr(sys, 'frozen', False):
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not base:
            base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path) if relative_path else base


BASE_DIR = writable_path()
MODEL_PATH = resource_path("modelo_comidas_peruanas.tflite")
CLASS_DIR = resource_path("comidas")
HISTORIAL_JSON = os.path.join(BASE_DIR, "historial_predicciones.json")

IMG_SIZE = (224, 224)

# >>> FIX: Orden fijo de clases, coincide EXACTAMENTE con las carpetas en C:\Proy4\comidas
# >>> y con el orden de entrenamiento del modelo.
ORDEN_CLASES_ENTRENAMIENTO = [
    "anticucho_peruano",
    "arroz_chaufa",
    "causa_limeña",
    "ceviche_peruano",
    "fideos_verdes_peruanos",
    "lomo_saltado",
    "pachamanca",
    "papa_rellena_peruana",
    "pizza",
    "pollo_a_la_brasa",
]

if os.path.isdir(CLASS_DIR):
    carpetas_en_disco = sorted(
        d for d in os.listdir(CLASS_DIR)
        if os.path.isdir(os.path.join(CLASS_DIR, d))
    )
    if set(carpetas_en_disco) == set(ORDEN_CLASES_ENTRENAMIENTO):
        CLASS_NAMES = list(ORDEN_CLASES_ENTRENAMIENTO)
    else:
        CLASS_NAMES = [c for c in ORDEN_CLASES_ENTRENAMIENTO if c in carpetas_en_disco]
        for c in carpetas_en_disco:
            if c not in CLASS_NAMES:
                CLASS_NAMES.append(c)
else:
    CLASS_NAMES = list(ORDEN_CLASES_ENTRENAMIENTO)


# ------------------------------------------------------------------
# Carga del modelo TFLite
# ------------------------------------------------------------------
def _cargar_interprete():
    if tflite is None:
        _abortar_sin_tflite()
    if not os.path.exists(MODEL_PATH):
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(
            None, "Modelo no encontrado",
            f"No se encontró el modelo:\n{MODEL_PATH}\n\n"
            "Coloca 'modelo_comidas_peruanas.tflite' junto al script."
        )
        sys.exit(1)

    interp = tflite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp


interpreter = _cargar_interprete()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def _inferencia(array_rgb):
    """Ejecuta la inferencia y devuelve el vector de salida crudo."""
    img = Image.fromarray(array_rgb).convert('RGB').resize(IMG_SIZE)

    if input_details[0]['dtype'] == np.uint8:
        array = np.array(img).astype(np.uint8)
    else:
        array = np.array(img).astype(np.float32) / 255.0
        array = array.astype(input_details[0]['dtype'])

    array = np.expand_dims(array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output


def _mapear_resultados(output, top_k=3):
    """
    Mapea el vector de salida a nombres de clase.
    Detecta si son probabilidades (softmax) o logits y aplica softmax si hace falta.
    """
    output = np.asarray(output, dtype=np.float32)
    suma = float(np.sum(output))

    if 0.99 < suma < 1.01 and np.all(output >= 0):
        probs = output
    else:
        exp = np.exp(output - np.max(output))
        probs = exp / np.sum(exp)

    n = min(top_k, len(CLASS_NAMES), len(probs))
    top_idx = np.argsort(probs)[-n:][::-1]

    resultados = []
    for i in top_idx:
        if i < len(CLASS_NAMES):
            resultados.append((CLASS_NAMES[i], float(probs[i]) * 100))
    return resultados


def predecir_imagen(image_path, top_k=3):
    t0 = time.perf_counter()
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img)
    output = _inferencia(arr)
    resultados = _mapear_resultados(output, top_k=top_k)
    tiempo_ms = (time.perf_counter() - t0) * 1000
    return resultados, tiempo_ms


def predecir_array(arr_rgb, top_k=3):
    t0 = time.perf_counter()
    output = _inferencia(arr_rgb)
    resultados = _mapear_resultados(output, top_k=top_k)
    tiempo_ms = (time.perf_counter() - t0) * 1000
    return resultados, tiempo_ms


# ------------------------------------------------------------------
# Colores según confianza
# ------------------------------------------------------------------
COLOR_VERDE = "#1e7e34"
COLOR_VERDE_BG = "#e6f4ea"
COLOR_AMARILLO = "#b8860b"
COLOR_AMARILLO_BG = "#fff8e1"
COLOR_ROJO = "#c0392b"
COLOR_ROJO_BG = "#fdecea"


def color_por_confianza(confianza):
    if confianza >= 80.0:
        return COLOR_VERDE, COLOR_VERDE_BG, "Predicción sólida"
    elif confianza >= 50.0:
        return COLOR_AMARILLO, COLOR_AMARILLO_BG, "Predicción dudosa"
    else:
        return COLOR_ROJO, COLOR_ROJO_BG, "Baja confianza"


NOMBRES_AMIGABLES = {
    "anticucho_peruano": "Anticucho peruano",
    "arroz_chaufa": "Arroz chaufa",
    "causa_limeña": "Causa limeña",
    "ceviche_peruano": "Ceviche peruano",
    "fideos_verdes_peruanos": "Fideos verdes peruanos",
    "lomo_saltado": "Lomo saltado",
    "pachamanca": "Pachamanca",
    "papa_rellena_peruana": "Papa rellena peruana",
    "pizza": "Pizza",
    "pollo_a_la_brasa": "Pollo a la brasa",
    "otros": "Otros (no comida)",
}


def nombre_amigable(clase):
    return NOMBRES_AMIGABLES.get(clase, clase.replace("_", " ").capitalize())


# ------------------------------------------------------------------
# Zona de drop personalizada
# ------------------------------------------------------------------
class ImageDropZone(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(240, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setText("")
        self._pixmap = None
        self._hover = False
        self.setAcceptDrops(True)

    def set_imagen(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def clear_imagen(self):
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = 12

        if self._pixmap:
            painter.setBrush(QColor("#ffffff"))
        else:
            painter.setBrush(QColor("#f7f9fc"))

        if self._hover:
            pen = QPen(QColor("#1a3a6b"), 2, Qt.SolidLine)
        elif self._pixmap:
            pen = QPen(QColor("#1a3a6b"), 2, Qt.SolidLine)
        else:
            pen = QPen(QColor("#c8d4e6"), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, radius, radius)

        if self._pixmap:
            margen = 8
            area = rect.adjusted(margen, margen, -margen, -margen)
            scaled = self._pixmap.scaled(
                area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = area.x() + (area.width() - scaled.width()) // 2
            y = area.y() + (area.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor("#8a94a3"))
            font = QFont("Segoe UI", 11)
            painter.setFont(font)
            painter.drawText(
                rect.adjusted(15, 0, -15, -20),
                Qt.AlignCenter | Qt.TextWordWrap,
                "Arrastra una imagen aquí"
            )
            font2 = QFont("Segoe UI", 8)
            painter.setFont(font2)
            painter.setPen(QColor("#b0b8c4"))
            painter.drawText(
                rect.adjusted(15, 25, -15, 0),
                Qt.AlignCenter | Qt.TextWordWrap,
                "o usa el botón de abajo\n(JPG, PNG, BMP)"
            )

        painter.end()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self._hover = True
            self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._hover = False
        self.update()

    def dropEvent(self, event):
        self._hover = False
        self.update()
        event.acceptProposedAction()


# ------------------------------------------------------------------
# Estilos
# ------------------------------------------------------------------
TEMA_CLARO = """
    QWidget {
        background: #eef1f7;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #1f2a3c;
    }
    QLabel { font-size: 12px; }

    QGroupBox {
        background: #ffffff;
        border: 1px solid #d4dce8;
        border-radius: 10px;
        margin-top: 14px;
        padding: 14px 12px 12px 12px;
        font-weight: bold;
        color: #1a3a6b;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        font-size: 10px;
        letter-spacing: 1px;
    }

    QPushButton {
        background-color: #ffffff;
        color: #1a3a6b;
        border: 1px solid #c8d4e6;
        border-radius: 7px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 500;
        min-height: 32px;
    }
    QPushButton:hover {
        background-color: #e4ecf7;
        border-color: #1a3a6b;
    }
    QPushButton:pressed { background-color: #d2dded; }
    QPushButton:disabled {
        background-color: #f3f5f8;
        color: #b0b8c4;
        border-color: #e0e5ec;
    }

    QProgressBar {
        border: 1px solid #c8d4e6;
        border-radius: 8px;
        background: #ffffff;
        height: 16px;
        text-align: center;
        font-size: 10px;
        color: #1a3a6b;
        font-weight: bold;
    }
    QProgressBar::chunk {
        background-color: #2f5fa0;
        border-radius: 8px;
    }

    QTableWidget {
        border: 1px solid #d4dce8;
        border-radius: 8px;
        background: #ffffff;
        font-size: 12px;
        gridline-color: #eef2f9;
        alternate-background-color: #f7f9fc;
    }
    QHeaderView::section {
        background: #1a3a6b;
        color: #ffffff;
        font-weight: bold;
        font-size: 11px;
        border: none;
        padding: 6px;
    }
    QComboBox {
        border: 1px solid #c8d4e6;
        border-radius: 6px;
        padding: 5px 8px;
        font-size: 12px;
        background: #ffffff;
    }
    QComboBox:hover { border-color: #1a3a6b; }

    QScrollArea { border: none; background: transparent; }

    QScrollBar:vertical {
        background: #eef2f9;
        width: 8px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #c8d4e6;
        border-radius: 4px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover { background: #8a94a3; }
"""

TEMA_OSCURO = """
    QWidget {
        background: #1a1f2b;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #d8dee9;
    }
    QLabel { font-size: 12px; color: #d8dee9; }

    QGroupBox {
        background: #232a38;
        border: 1px solid #3b465c;
        border-radius: 10px;
        margin-top: 14px;
        padding: 14px 12px 12px 12px;
        font-weight: bold;
        color: #8ab4e8;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        font-size: 10px;
        letter-spacing: 1px;
    }

    QPushButton {
        background-color: #2a3345;
        color: #d8dee9;
        border: 1px solid #3b465c;
        border-radius: 7px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 500;
        min-height: 32px;
    }
    QPushButton:hover { background-color: #364159; border-color: #5b7bb5; }
    QPushButton:pressed { background-color: #2a3345; }
    QPushButton:disabled {
        background-color: #1f2632;
        color: #5b6478;
        border-color: #2f3747;
    }

    QProgressBar {
        border: 1px solid #3b465c;
        border-radius: 8px;
        background: #2a3345;
        height: 16px;
        text-align: center;
        font-size: 10px;
        color: #d8dee9;
        font-weight: bold;
    }
    QProgressBar::chunk {
        background-color: #4a90d9;
        border-radius: 8px;
    }

    QTableWidget {
        border: 1px solid #3b465c;
        border-radius: 8px;
        background: #232a38;
        color: #d8dee9;
        font-size: 12px;
        gridline-color: #3b465c;
        alternate-background-color: #2a3345;
    }
    QHeaderView::section {
        background: #2a3345;
        color: #d8dee9;
        font-weight: bold;
        font-size: 11px;
        border: none;
        padding: 6px;
    }
    QComboBox {
        border: 1px solid #3b465c;
        border-radius: 6px;
        padding: 5px 8px;
        font-size: 12px;
        background: #2a3345;
        color: #d8dee9;
    }
    QComboBox:hover { border-color: #5b7bb5; }

    QScrollArea { border: none; background: transparent; }

    QScrollBar:vertical {
        background: #1f2632;
        width: 8px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #3b465c;
        border-radius: 4px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover { background: #5b7bb5; }
"""


# ------------------------------------------------------------------
# Toast
# ------------------------------------------------------------------
class Toast(QLabel):
    def __init__(self, parent, mensaje, duracion=2500):
        super().__init__(mensaje, parent)
        self.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: #ffffff;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 12px;
                font-weight: 500;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.adjustSize()

        x = (parent.width() - self.width()) // 2
        y = parent.height() - self.height() - 30
        self.move(x, y)
        self.show()
        self.raise_()

        self._efecto = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._efecto)
        self._anim = QPropertyAnimation(self._efecto, b"opacity")
        self._anim.setDuration(duracion)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim.finished.connect(self.close)
        self._anim.start()


# ------------------------------------------------------------------
# Hilo para captura de webcam
# ------------------------------------------------------------------
class WebcamThread(QThread):
    frame_ready = None

    def __init__(self, camara_index=0):
        super().__init__()
        self.camara_index = camara_index
        self._running = False

    def run(self):
        if not _HAS_CV2:
            return
        cap = cv2.VideoCapture(self.camara_index)
        if not cap.isOpened():
            return
        self._running = True
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_ready is not None:
                self.frame_ready(frame)
            self.msleep(50)
        cap.release()

    def stop(self):
        self._running = False
        self.wait(1000)


# ------------------------------------------------------------------
# Ventana de Webcam
# ------------------------------------------------------------------
class WebcamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clasificación en Vivo (Webcam)")
        self.resize(680, 600)
        self.parent_app = parent
        self.thread = None
        self.ultimo_frame = None
        self.contador_pred = 0

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        self.label_video = QLabel("Iniciando cámara...")
        self.label_video.setFixedSize(600, 440)
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setStyleSheet(
            "border: 2px solid #c8d4e6; border-radius: 10px; "
            "background: #f0f2f5; color: #8a94a3; font-size: 13px;"
        )
        layout.addWidget(self.label_video, alignment=Qt.AlignCenter)

        self.label_resultado = QLabel("Esperando predicción...")
        self.label_resultado.setAlignment(Qt.AlignCenter)
        self.label_resultado.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1a3a6b; "
            "padding: 10px; background: #eef2f9; border-radius: 8px; "
            "border: 1px solid #d6deeb;"
        )
        self.label_resultado.setMinimumHeight(55)
        self.label_resultado.setWordWrap(True)
        layout.addWidget(self.label_resultado)

        botones = QHBoxLayout()
        botones.setSpacing(10)

        self.btn_capturar = QPushButton("Clasificar Ahora")
        self.btn_capturar.setMinimumHeight(38)
        self.btn_capturar.setCursor(Qt.PointingHandCursor)
        self.btn_capturar.clicked.connect(self.clasificar_frame)
        botones.addWidget(self.btn_capturar)

        self.btn_detener = QPushButton("Detener Cámara")
        self.btn_detener.setMinimumHeight(38)
        self.btn_detener.setCursor(Qt.PointingHandCursor)
        self.btn_detener.clicked.connect(self.detener)
        botones.addWidget(self.btn_detener)

        self.btn_cerrar = QPushButton("Cerrar")
        self.btn_cerrar.setMinimumHeight(38)
        self.btn_cerrar.setCursor(Qt.PointingHandCursor)
        self.btn_cerrar.clicked.connect(self.close)
        botones.addWidget(self.btn_cerrar)

        layout.addLayout(botones)
        self.setLayout(layout)

        if not _HAS_CV2:
            self.label_video.setText(
                "OpenCV no está instalado.\n\n"
                "Instala con:\n"
                "python -m pip install opencv-python"
            )
            self.btn_capturar.setEnabled(False)
            self.btn_detener.setEnabled(False)
            return

        self._iniciar_camara()

    def _iniciar_camara(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            self.label_video.setText(
                "No se detectó ninguna cámara.\n\n"
                "Conecta una webcam y vuelve a intentarlo."
            )
            self.btn_capturar.setEnabled(False)
            self.btn_detener.setEnabled(False)
            return
        cap.release()

        self.thread = WebcamThread(0)
        self.thread.frame_ready = self._on_frame
        self.thread.start()

    def _on_frame(self, frame):
        self.ultimo_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            600, 440, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_video.setPixmap(pixmap)

        self.contador_pred += 1
        if self.contador_pred % 10 == 0:
            self._clasificar_frame_actual(rgb)

    def _clasificar_frame_actual(self, rgb):
        try:
            top3, tiempo = predecir_array(rgb, top_k=1)
            if not top3:
                return
            clase, conf = top3[0]
            color_txt, color_bg, etiqueta = color_por_confianza(conf)
            self.label_resultado.setText(
                f"{nombre_amigable(clase)}  —  {conf:.1f}%   ({etiqueta})   "
                f"[{tiempo:.0f} ms]"
            )
            self.label_resultado.setStyleSheet(
                f"font-size: 14px; font-weight: bold; color: {color_txt}; "
                f"padding: 10px; background: {color_bg}; border-radius: 8px; "
                f"border: 1px solid {color_txt};"
            )
        except Exception:
            pass

    def clasificar_frame(self):
        if self.ultimo_frame is None:
            self.label_resultado.setText("No hay frame disponible.")
            return
        rgb = cv2.cvtColor(self.ultimo_frame, cv2.COLOR_BGR2RGB)
        try:
            top3, tiempo = predecir_array(rgb, top_k=3)
            if not top3:
                return
            clase, conf = top3[0]
            color_txt, color_bg, etiqueta = color_por_confianza(conf)
            self.label_resultado.setText(
                f"{nombre_amigable(clase)}  —  {conf:.1f}%   ({etiqueta})   "
                f"[{tiempo:.0f} ms]"
            )
            self.label_resultado.setStyleSheet(
                f"font-size: 14px; font-weight: bold; color: {color_txt}; "
                f"padding: 10px; background: {color_bg}; border-radius: 8px; "
                f"border: 1px solid {color_txt};"
            )
        except Exception as e:
            self.label_resultado.setText(f"Error: {e}")

    def detener(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.label_video.setText("Cámara detenida.")
        self.btn_capturar.setEnabled(False)
        self.btn_detener.setEnabled(False)

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
            self.thread = None
        event.accept()


# ------------------------------------------------------------------
# Ventana principal
# ------------------------------------------------------------------
class ClasificadorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificador de Comidas Peruanas")
        self.resize(1100, 720)
        self.setMinimumSize(1050, 680)

        self.tema_oscuro = False
        self.historial = []          # >>> FIX: siempre empieza vacío
        self._borrar_historial_archivo()  # >>> FIX: borra el .json viejo al arrancar

        # ==========================================================
        # TÍTULO
        # ==========================================================
        titulo = QLabel("Clasificador de Comidas Peruanas")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #1a3a6b; "
            "letter-spacing: 0.5px; padding: 4px 0 0 0;"
        )

        subtitulo = QLabel(
            "Sistema de reconocimiento de platos típicos con Inteligencia Artificial"
        )
        subtitulo.setAlignment(Qt.AlignCenter)
        subtitulo.setStyleSheet(
            "font-size: 11px; color: #8a94a3; padding-bottom: 6px;"
        )

        # ==========================================================
        # PANEL IZQUIERDO
        # ==========================================================
        panel_izq = QVBoxLayout()
        panel_izq.setSpacing(10)

        grupo_imagen = QGroupBox("IMAGEN DE ENTRADA")
        layout_imagen = QVBoxLayout()
        layout_imagen.setContentsMargins(12, 12, 12, 12)
        layout_imagen.setSpacing(10)

        self.drop_zone = ImageDropZone()
        layout_imagen.addWidget(self.drop_zone, alignment=Qt.AlignCenter)

        self.btn_seleccionar = QPushButton("Seleccionar Imagen")
        self.btn_seleccionar.setMinimumHeight(38)
        self.btn_seleccionar.setCursor(Qt.PointingHandCursor)
        self.btn_seleccionar.setStyleSheet("""
            QPushButton {
                background-color: #1a3a6b;
                color: #ffffff;
                border-radius: 7px;
                padding: 8px 14px;
                font-size: 13px;
                font-weight: bold;
                border: none;
                min-height: 38px;
            }
            QPushButton:hover { background-color: #234b85; }
            QPushButton:pressed { background-color: #142c52; }
        """)
        self.btn_seleccionar.clicked.connect(self.seleccionar_imagen)
        layout_imagen.addWidget(self.btn_seleccionar)

        grupo_imagen.setLayout(layout_imagen)
        panel_izq.addWidget(grupo_imagen)

        grupo_herramientas = QGroupBox("HERRAMIENTAS")
        layout_herr = QGridLayout()
        layout_herr.setContentsMargins(12, 12, 12, 12)
        layout_herr.setHorizontalSpacing(8)
        layout_herr.setVerticalSpacing(8)

        self.btn_webcam = QPushButton("Webcam en Vivo")
        self.btn_webcam.setCursor(Qt.PointingHandCursor)
        self.btn_webcam.clicked.connect(self.abrir_webcam)
        layout_herr.addWidget(self.btn_webcam, 0, 0)

        self.btn_demo = QPushButton("Modo Demo")
        self.btn_demo.setCursor(Qt.PointingHandCursor)
        self.btn_demo.clicked.connect(self.modo_demo)
        layout_herr.addWidget(self.btn_demo, 0, 1)

        self.btn_comparar = QPushButton("Comparar Imágenes")
        self.btn_comparar.setCursor(Qt.PointingHandCursor)
        self.btn_comparar.clicked.connect(self.comparar_imagenes)
        layout_herr.addWidget(self.btn_comparar, 1, 0)

        self.btn_galeria = QPushButton("Galería de Ejemplos")
        self.btn_galeria.setCursor(Qt.PointingHandCursor)
        self.btn_galeria.clicked.connect(self.mostrar_galeria)
        layout_herr.addWidget(self.btn_galeria, 1, 1)

        layout_herr.setColumnStretch(0, 1)
        layout_herr.setColumnStretch(1, 1)

        grupo_herramientas.setLayout(layout_herr)
        panel_izq.addWidget(grupo_herramientas)

        grupo_resultado = QGroupBox("RESULTADO DE LA PREDICCIÓN")
        layout_resultado = QVBoxLayout()
        layout_resultado.setContentsMargins(12, 12, 12, 12)
        layout_resultado.setSpacing(8)

        self.label_resultado = QLabel("Predicción: —")
        self.label_resultado.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #1a3a6b; "
            "padding: 12px; background: #eef2f9; border-radius: 7px; "
            "border: 1px solid #d6deeb;"
        )
        self.label_resultado.setWordWrap(True)
        self.label_resultado.setAlignment(Qt.AlignCenter)
        self.label_resultado.setMinimumHeight(70)
        layout_resultado.addWidget(self.label_resultado)

        self.label_tiempo = QLabel("Tiempo de inferencia: —")
        self.label_tiempo.setStyleSheet(
            "font-size: 10px; color: #8a94a3; padding: 2px;"
        )
        self.label_tiempo.setAlignment(Qt.AlignCenter)
        layout_resultado.addWidget(self.label_tiempo)

        grupo_resultado.setLayout(layout_resultado)
        panel_izq.addWidget(grupo_resultado)

        grupo_top3 = QGroupBox("TOP 3 PREDICCIONES")
        layout_top3 = QVBoxLayout()
        layout_top3.setContentsMargins(12, 12, 12, 12)
        layout_top3.setSpacing(8)

        self.top_widgets = []
        for _ in range(3):
            fila = QHBoxLayout()
            fila.setSpacing(8)

            lbl_nombre = QLabel("—")
            lbl_nombre.setFixedWidth(150)
            lbl_nombre.setStyleSheet(
                "font-size: 11px; color: #2c3e50; font-weight: 500;"
            )

            barra = QProgressBar()
            barra.setMinimum(0)
            barra.setMaximum(100)
            barra.setValue(0)
            barra.setFormat("%p%")
            barra.setFixedHeight(18)
            barra.setTextVisible(True)

            fila.addWidget(lbl_nombre)
            fila.addWidget(barra, 1)

            contenedor = QWidget()
            contenedor.setLayout(fila)
            layout_top3.addWidget(contenedor)
            self.top_widgets.append((lbl_nombre, barra, contenedor))

        grupo_top3.setLayout(layout_top3)
        panel_izq.addWidget(grupo_top3)

        panel_izq.addStretch()

        # ==========================================================
        # PANEL DERECHO
        # ==========================================================
        panel_der = QVBoxLayout()
        panel_der.setSpacing(10)

        grupo_platos = QGroupBox("PLATOS RECONOCIDOS")
        layout_platos = QVBoxLayout()
        layout_platos.setContentsMargins(12, 12, 12, 12)
        layout_platos.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(150)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        contenedor_platos = QWidget()
        contenedor_platos.setStyleSheet("background: transparent;")
        grid = QGridLayout()
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        for i, clase in enumerate(CLASS_NAMES):
            chip = QLabel(nombre_amigable(clase))
            chip.setStyleSheet(
                "background: #eef2f9; color: #1a3a6b; "
                "border: 1px solid #c8d4e6; border-radius: 12px; "
                "padding: 6px 10px; font-size: 11px; font-weight: 500;"
            )
            chip.setAlignment(Qt.AlignCenter)
            chip.setMinimumHeight(30)
            grid.addWidget(chip, i // 3, i % 3)

        for c in range(3):
            grid.setColumnStretch(c, 1)

        contenedor_platos.setLayout(grid)
        scroll.setWidget(contenedor_platos)
        layout_platos.addWidget(scroll)
        grupo_platos.setLayout(layout_platos)
        panel_der.addWidget(grupo_platos)

        grupo_metricas = QGroupBox("MÉTRICAS DE LA SESIÓN")
        layout_metricas = QVBoxLayout()
        layout_metricas.setContentsMargins(12, 12, 12, 12)
        layout_metricas.setSpacing(8)

        self.label_metricas = QLabel(
            "Imágenes: 0    Aciertos: 0    Correcciones: 0    Errores: 0"
        )
        self.label_metricas.setStyleSheet(
            "font-size: 11px; color: #555; padding: 8px; "
            "background: #eef2f9; border-radius: 6px;"
        )
        self.label_metricas.setWordWrap(True)
        self.label_metricas.setAlignment(Qt.AlignCenter)
        layout_metricas.addWidget(self.label_metricas)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setFormat("Porcentaje de acierto: %p%")
        self.progress.setFixedHeight(20)
        layout_metricas.addWidget(self.progress)

        grupo_metricas.setLayout(layout_metricas)
        panel_der.addWidget(grupo_metricas)

        grupo_grafico = QGroupBox("PREDICCIÓN ACTUAL")
        layout_grafico = QVBoxLayout()
        layout_grafico.setContentsMargins(12, 12, 12, 12)

        self.fig, self.ax = plt.subplots(figsize=(4.5, 2.2))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedHeight(180)
        self.canvas.setStyleSheet("background: transparent;")
        layout_grafico.addWidget(self.canvas)

        grupo_grafico.setLayout(layout_grafico)
        panel_der.addWidget(grupo_grafico)

        grupo_acciones = QGroupBox("ACCIONES")
        layout_acciones = QVBoxLayout()
        layout_acciones.setContentsMargins(12, 12, 12, 12)
        layout_acciones.setSpacing(8)

        self.btn_feedback = QPushButton("La predicción es correcta")
        self.btn_feedback.clicked.connect(self.dar_feedback)
        self.btn_feedback.setEnabled(False)
        self.btn_feedback.setMinimumHeight(36)
        self.btn_feedback.setCursor(Qt.PointingHandCursor)
        layout_acciones.addWidget(self.btn_feedback)

        fila_acciones = QHBoxLayout()
        fila_acciones.setSpacing(8)

        self.btn_historial = QPushButton("Ver historial")
        self.btn_historial.clicked.connect(self.mostrar_historial)
        self.btn_historial.setMinimumHeight(36)
        self.btn_historial.setCursor(Qt.PointingHandCursor)
        fila_acciones.addWidget(self.btn_historial)

        self.btn_pdf = QPushButton("Exportar PDF")
        self.btn_pdf.clicked.connect(self.exportar_pdf)
        self.btn_pdf.setMinimumHeight(36)
        self.btn_pdf.setCursor(Qt.PointingHandCursor)
        fila_acciones.addWidget(self.btn_pdf)

        self.btn_exportar = QPushButton("Exportar CSV")
        self.btn_exportar.clicked.connect(self.exportar_csv)
        self.btn_exportar.setMinimumHeight(36)
        self.btn_exportar.setCursor(Qt.PointingHandCursor)
        fila_acciones.addWidget(self.btn_exportar)

        layout_acciones.addLayout(fila_acciones)

        fila_acciones2 = QHBoxLayout()
        fila_acciones2.setSpacing(8)

        self.btn_limpiar = QPushButton("Limpiar historial")
        self.btn_limpiar.clicked.connect(self.limpiar_historial)
        self.btn_limpiar.setMinimumHeight(36)
        self.btn_limpiar.setCursor(Qt.PointingHandCursor)
        fila_acciones2.addWidget(self.btn_limpiar)

        self.btn_tema = QPushButton("Modo Oscuro")
        self.btn_tema.clicked.connect(self.alternar_tema)
        self.btn_tema.setMinimumHeight(36)
        self.btn_tema.setCursor(Qt.PointingHandCursor)
        fila_acciones2.addWidget(self.btn_tema)

        layout_acciones.addLayout(fila_acciones2)

        grupo_acciones.setLayout(layout_acciones)
        panel_der.addWidget(grupo_acciones)

        panel_der.addStretch()

        # ==========================================================
        # LAYOUT PRINCIPAL
        # ==========================================================
        layout_principal = QHBoxLayout()
        layout_principal.setContentsMargins(15, 6, 15, 12)
        layout_principal.setSpacing(14)
        layout_principal.addLayout(panel_izq, 42)
        layout_principal.addLayout(panel_der, 58)

        contenedor = QVBoxLayout()
        contenedor.setContentsMargins(0, 0, 0, 0)
        contenedor.setSpacing(2)
        contenedor.addWidget(titulo)
        contenedor.addWidget(subtitulo)
        contenedor.addLayout(layout_principal)

        self.setLayout(contenedor)
        self.setStyleSheet(TEMA_CLARO)

        self.ultima_imagen_path = None
        self.ultima_prediccion = None
        self.ultima_confianza = 0.0
        self.class_names = list(CLASS_NAMES)
        self.demo_activo = False
        self._demo_cola = []
        self._demo_idx = 0

        self.actualizar_metricas()

    # --------------------------------------------------------------
    def _borrar_historial_archivo(self):
        """Elimina el archivo .json del historial si existe."""
        try:
            if os.path.exists(HISTORIAL_JSON):
                os.remove(HISTORIAL_JSON)
        except Exception:
            pass

    def _guardar_historial(self):
        try:
            with open(HISTORIAL_JSON, 'w', encoding='utf-8') as f:
                json.dump(self.historial, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # --------------------------------------------------------------
    def alternar_tema(self):
        self.tema_oscuro = not self.tema_oscuro
        if self.tema_oscuro:
            self.setStyleSheet(TEMA_OSCURO)
            self.btn_tema.setText("Modo Claro")
        else:
            self.setStyleSheet(TEMA_CLARO)
            self.btn_tema.setText("Modo Oscuro")
        self._actualizar_grafico_tema()

    def _actualizar_grafico_tema(self):
        if self.tema_oscuro:
            self.fig.patch.set_facecolor('#232a38')
            self.ax.set_facecolor('#232a38')
            self.ax.tick_params(colors='#d8dee9')
            for spine in self.ax.spines.values():
                spine.set_color('#3b465c')
        else:
            self.fig.patch.set_facecolor('#ffffff')
            self.ax.set_facecolor('#ffffff')
            self.ax.tick_params(colors='#555')
            for spine in self.ax.spines.values():
                spine.set_color('#cccccc')
        self.actualizar_metricas()

    def toast(self, mensaje):
        Toast(self, mensaje)

    # --------------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            ruta = url.toLocalFile()
            if ruta.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.procesar_imagen(ruta)
                break

    # --------------------------------------------------------------
    def procesar_imagen(self, ruta):
        try:
            img = Image.open(ruta).convert('RGB').resize(IMG_SIZE)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir la imagen:\n{e}")
            return

        arr_img = np.array(img)
        h, w, ch = arr_img.shape
        bytes_per_line = ch * w
        qimg = QImage(arr_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.drop_zone.set_imagen(pixmap)

        try:
            top3, tiempo_ms = predecir_imagen(ruta, top_k=3)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo predecir la imagen:\n{e}")
            return

        if not top3:
            return

        self.label_tiempo.setText(f"Tiempo de inferencia: {tiempo_ms:.1f} ms")

        clase_top, confianza_top = top3[0]
        color_txt, color_bg, etiqueta = color_por_confianza(confianza_top)

        for i, (lbl_nombre, barra, _) in enumerate(self.top_widgets):
            if i < len(top3):
                nombre, conf = top3[i]
                lbl_nombre.setText(nombre_amigable(nombre))
                barra.setValue(int(conf))
                if i == 0:
                    barra.setStyleSheet(f"""
                        QProgressBar {{
                            border: 1px solid #c8d4e6;
                            border-radius: 8px;
                            background: #ffffff;
                            text-align: center;
                            font-size: 10px;
                            color: {color_txt};
                            font-weight: bold;
                        }}
                        QProgressBar::chunk {{
                            background-color: {color_txt};
                            border-radius: 8px;
                        }}
                    """)
                else:
                    barra.setStyleSheet("""
                        QProgressBar {
                            border: 1px solid #c8d4e6;
                            border-radius: 8px;
                            background: #ffffff;
                            text-align: center;
                            font-size: 10px;
                            color: #1a3a6b;
                            font-weight: bold;
                        }
                        QProgressBar::chunk {
                            background-color: #8ab4e8;
                            border-radius: 8px;
                        }
                    """)
            else:
                lbl_nombre.setText("—")
                barra.setValue(0)

        if confianza_top <= 35.0:
            self.label_resultado.setText(
                f"No se reconoce ninguna clase\n"
                f"Confianza: {confianza_top:.2f}%"
            )
            self.label_resultado.setStyleSheet(
                f"font-size: 13px; font-weight: bold; color: {COLOR_ROJO}; "
                f"padding: 12px; background: {COLOR_ROJO_BG}; "
                f"border-radius: 7px; border: 1px solid {COLOR_ROJO};"
            )
            self.ultima_imagen_path = ruta
            self.ultima_prediccion = None
            self.ultima_confianza = confianza_top
            self.btn_feedback.setEnabled(False)
            self.historial.append([ruta, "Ninguna", confianza_top, "", ""])
            self._guardar_historial()
            self.actualizar_metricas()
            if not self.demo_activo:
                self.agregar_nueva_clase(ruta)
        else:
            self.label_resultado.setText(
                f"{nombre_amigable(clase_top)}\n"
                f"Confianza: {confianza_top:.2f}%  |  {etiqueta}"
            )
            self.label_resultado.setStyleSheet(
                f"font-size: 13px; font-weight: bold; color: {color_txt}; "
                f"padding: 12px; background: {color_bg}; "
                f"border-radius: 7px; border: 1px solid {color_txt};"
            )
            self.ultima_imagen_path = ruta
            self.ultima_prediccion = clase_top
            self.ultima_confianza = confianza_top
            self.btn_feedback.setEnabled(True)
            self.historial.append([ruta, clase_top, confianza_top, "", ""])
            self._guardar_historial()
            self.actualizar_metricas()

    # --------------------------------------------------------------
    def seleccionar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if not ruta:
            return
        self.procesar_imagen(ruta)

    # --------------------------------------------------------------
    def abrir_webcam(self):
        dlg = WebcamDialog(self)
        dlg.exec_()

    # --------------------------------------------------------------
    def modo_demo(self):
        if not os.path.isdir(CLASS_DIR):
            self.toast("No se encontró la carpeta 'comidas'.")
            return

        imagenes = []
        for clase in CLASS_NAMES:
            ruta_clase = os.path.join(CLASS_DIR, clase)
            if not os.path.isdir(ruta_clase):
                continue
            for f in os.listdir(ruta_clase):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    imagenes.append(os.path.join(ruta_clase, f))

        if not imagenes:
            self.toast("No hay imágenes en 'comidas/' para la demo.")
            return

        muestra = random.sample(imagenes, min(5, len(imagenes)))
        self.demo_activo = True
        self._demo_cola = muestra
        self._demo_idx = 0
        self.toast(f"Modo demostración: {len(muestra)} imágenes")
        QTimer.singleShot(800, self._demo_siguiente)

    def _demo_siguiente(self):
        if self._demo_idx >= len(self._demo_cola):
            self.demo_activo = False
            self.toast("Demostración finalizada")
            return

        ruta = self._demo_cola[self._demo_idx]
        self._demo_idx += 1
        self.procesar_imagen(ruta)
        QTimer.singleShot(2000, self._demo_siguiente)

    # --------------------------------------------------------------
    def comparar_imagenes(self):
        ruta1, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar primera imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if not ruta1:
            return
        ruta2, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar segunda imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if not ruta2:
            return

        try:
            top1, t1 = predecir_imagen(ruta1, top_k=1)
            top2, t2 = predecir_imagen(ruta2, top_k=1)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo comparar:\n{e}")
            return

        clase1, conf1 = top1[0]
        clase2, conf2 = top2[0]

        dialog = QDialog(self)
        dialog.setWindowTitle("Comparación de Imágenes")
        dialog.resize(680, 420)

        layout = QHBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        for ruta, clase, conf, tiempo in [(ruta1, clase1, conf1, t1),
                                           (ruta2, clase2, conf2, t2)]:
            col = QVBoxLayout()
            img = Image.open(ruta).convert('RGB').resize((260, 260))
            arr = np.array(img)
            h, w, ch = arr.shape
            qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
            lbl_img = QLabel()
            lbl_img.setPixmap(QPixmap.fromImage(qimg))
            lbl_img.setAlignment(Qt.AlignCenter)
            col.addWidget(lbl_img)

            color_txt, color_bg, etiqueta = color_por_confianza(conf)
            lbl_info = QLabel(
                f"{nombre_amigable(clase)}\n"
                f"Confianza: {conf:.2f}%\n"
                f"Tiempo: {tiempo:.1f} ms\n"
                f"{etiqueta}"
            )
            lbl_info.setAlignment(Qt.AlignCenter)
            lbl_info.setStyleSheet(
                f"font-size: 12px; font-weight: bold; color: {color_txt}; "
                f"background: {color_bg}; border-radius: 8px; "
                f"border: 1px solid {color_txt}; padding: 10px;"
            )
            col.addWidget(lbl_info)
            layout.addLayout(col)

        dialog.setLayout(layout)
        dialog.exec_()

    # --------------------------------------------------------------
    def mostrar_galeria(self):
        if not os.path.isdir(CLASS_DIR):
            self.toast("No se encontró la carpeta 'comidas'.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Galería de Ejemplos por Clase")
        dialog.resize(820, 600)

        layout_principal = QVBoxLayout()
        layout_principal.setContentsMargins(15, 15, 15, 15)
        layout_principal.setSpacing(10)

        combo_clase = QComboBox()
        combo_clase.addItems([nombre_amigable(c) for c in CLASS_NAMES])
        layout_principal.addWidget(QLabel("Seleccione una clase:"))
        layout_principal.addWidget(combo_clase)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contenedor = QWidget()
        grid = QGridLayout()
        grid.setSpacing(8)
        contenedor.setLayout(grid)
        scroll.setWidget(contenedor)
        layout_principal.addWidget(scroll)

        def actualizar_galeria():
            idx = combo_clase.currentIndex()
            if idx < 0 or idx >= len(CLASS_NAMES):
                return
            clase = CLASS_NAMES[idx]
            ruta_clase = os.path.join(CLASS_DIR, clase)

            while grid.count():
                item = grid.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

            if not os.path.isdir(ruta_clase):
                grid.addWidget(QLabel("Carpeta no encontrada"), 0, 0)
                return

            imagenes = [f for f in os.listdir(ruta_clase)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if not imagenes:
                grid.addWidget(QLabel("No hay imágenes en esta clase"), 0, 0)
                return

            for i, nombre in enumerate(imagenes[:24]):
                ruta = os.path.join(ruta_clase, nombre)
                try:
                    img = Image.open(ruta).convert('RGB').resize((130, 130))
                    arr = np.array(img)
                    h, w, ch = arr.shape
                    qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
                    lbl = QLabel()
                    lbl.setPixmap(QPixmap.fromImage(qimg))
                    lbl.setStyleSheet("border: 1px solid #c8d4e6; border-radius: 6px;")
                    grid.addWidget(lbl, i // 4, i % 4)
                except Exception:
                    pass

        combo_clase.currentIndexChanged.connect(actualizar_galeria)
        actualizar_galeria()

        dialog.setLayout(layout_principal)
        dialog.exec_()

    # --------------------------------------------------------------
    def _guardar_imagen_en_clase(self, ruta_imagen, nueva_clase):
        if nueva_clase not in self.class_names:
            self.class_names.append(nueva_clase)

        ruta_guardado = writable_path(os.path.join("imagenes_corregidas", nueva_clase))
        os.makedirs(ruta_guardado, exist_ok=True)

        nombre_archivo = os.path.basename(ruta_imagen)
        i = 1
        nuevo_nombre = nombre_archivo
        while os.path.exists(os.path.join(ruta_guardado, nuevo_nombre)):
            base, ext = os.path.splitext(nombre_archivo)
            nuevo_nombre = f"{base}_{i}{ext}"
            i += 1

        destino = os.path.join(ruta_guardado, nuevo_nombre)
        shutil.copy(ruta_imagen, destino)
        return destino

    def agregar_nueva_clase(self, ruta_imagen):
        resp = QMessageBox.question(
            self, "Nueva clase",
            "¿Desea agregar una nueva clase con esta imagen?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        nueva_clase, ok = QInputDialog.getText(
            self, "Nueva clase", "Escriba el nombre de la nueva clase:"
        )
        if ok and nueva_clase.strip():
            self._guardar_imagen_en_clase(ruta_imagen, nueva_clase.strip())
            self.toast(f"Imagen guardada en '{nueva_clase}'")

    # --------------------------------------------------------------
    def dar_feedback(self):
        if self.ultima_imagen_path is None:
            return

        resp = QMessageBox.question(
            self, "Confirmar",
            f"¿La predicción ({nombre_amigable(self.ultima_prediccion)}) es correcta?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        index = len(self.historial) - 1

        if resp == QMessageBox.Yes:
            self.toast("Gracias por su confirmación")
            if index >= 0:
                ruta, prediccion, confianza, _, _ = self.historial[index]
                self.historial[index] = [ruta, prediccion, confianza, "Sí", prediccion]
                self._guardar_historial()
            self.actualizar_metricas()

        elif resp == QMessageBox.No:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Opciones")
            dialog.setText("¿Qué desea hacer con esta imagen?")
            btn_agregar = dialog.addButton("Agregar clase nueva", QMessageBox.ActionRole)
            btn_ninguna = dialog.addButton("No es ninguna", QMessageBox.ActionRole)
            dialog.addButton(QMessageBox.Cancel)
            dialog.exec_()
            selected = dialog.clickedButton()

            if selected == btn_agregar:
                nueva_clase, ok = QInputDialog.getText(
                    self, "Nueva clase", "Escriba el nombre de la nueva clase:"
                )
                if ok and nueva_clase.strip():
                    self._guardar_imagen_en_clase(
                        self.ultima_imagen_path, nueva_clase.strip()
                    )
                    self.toast(f"Imagen guardada en '{nueva_clase}'")
                    if index >= 0:
                        ruta, prediccion, confianza, _, _ = self.historial[index]
                        self.historial[index] = [ruta, prediccion, confianza, "No", nueva_clase.strip()]
                        self._guardar_historial()
                    self.actualizar_metricas()

            elif selected == btn_ninguna:
                if index >= 0:
                    ruta, prediccion, confianza, _, _ = self.historial[index]
                    self.historial[index] = [ruta, prediccion, confianza, "No", "Ninguna"]
                    self._guardar_historial()
                self.toast("Marcado como ninguna clase")
                self.actualizar_metricas()

    # --------------------------------------------------------------
    def limpiar_historial(self):
        """Borra el historial en memoria y en disco, y resetea la UI."""
        self.historial = []
        self._borrar_historial_archivo()
        self.ultima_prediccion = None
        self.ultima_imagen_path = None
        self.ultima_confianza = 0.0
        self.btn_feedback.setEnabled(False)
        self.drop_zone.clear_imagen()
        self.label_resultado.setText("Predicción: —")
        self.label_resultado.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #1a3a6b; "
            "padding: 12px; background: #eef2f9; border-radius: 7px; "
            "border: 1px solid #d6deeb;"
        )
        self.label_tiempo.setText("Tiempo de inferencia: —")
        for lbl_nombre, barra, _ in self.top_widgets:
            lbl_nombre.setText("—")
            barra.setValue(0)
        self.actualizar_metricas()
        self.toast("Historial limpiado")

    # --------------------------------------------------------------
    def actualizar_metricas(self):
        total = len(self.historial)
        aciertos = sum(1 for h in self.historial if h[3] == "Sí")
        correcciones = sum(1 for h in self.historial if h[3] == "No" and h[4] not in ("", "Ninguna"))
        errores_ninguna = sum(1 for h in self.historial if h[4] == "Ninguna")
        porcentaje = (aciertos / total * 100) if total > 0 else 0

        self.label_metricas.setText(
            f"Imágenes: {total}    Aciertos: {aciertos}    "
            f"Correcciones: {correcciones}    Errores: {errores_ninguna}"
        )
        self.progress.setValue(int(porcentaje))

        # >>> FIX: el gráfico muestra SOLO la predicción actual
        if self.ultima_prediccion:
            conteos = Counter([self.ultima_prediccion])
        elif total > 0 and self.historial[-1][1] == "Ninguna":
            conteos = Counter(["Ninguna"])
        else:
            conteos = Counter()

        clases = list(conteos.keys())
        valores = list(conteos.values())

        if self.tema_oscuro:
            colores = ["#e74c3c" if c == "Ninguna" else "#4a90d9" for c in clases]
            color_titulo = "#d8dee9"
            color_ejes = "#d8dee9"
        else:
            colores = ["#c0392b" if c == "Ninguna" else "#4a90d9" for c in clases]
            color_titulo = "#1a3a6b"
            color_ejes = "#555"

        self.ax.clear()
        if clases:
            etiquetas_mostrar = [nombre_amigable(c) for c in clases]
            self.ax.bar(etiquetas_mostrar, valores, color=colores,
                        edgecolor=color_ejes, linewidth=0.5)
            self.ax.set_ylabel("Cantidad", fontsize=8, color=color_ejes)
            self.ax.set_title("Predicción actual", fontsize=9,
                              color=color_titulo, fontweight="bold")
            self.ax.tick_params(axis='x', rotation=15, labelsize=8, colors=color_ejes)
            self.ax.tick_params(axis='y', labelsize=7, colors=color_ejes)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
        else:
            self.ax.text(0.5, 0.5, 'Aún sin datos', ha='center', va='center',
                         fontsize=9, color=color_ejes)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw()

    # --------------------------------------------------------------
    def mostrar_historial(self):
        class HistorialDialog(QDialog):
            def __init__(dlg_self, historial):
                super().__init__()
                dlg_self.setWindowTitle("Historial de predicciones")
                dlg_self.resize(880, 500)

                unique_classes = sorted(list(set(
                    [h[1] for h in historial if h[1] not in ["", "Ninguna"]]
                    + [h[4] for h in historial if h[4] not in ["", "Ninguna"]]
                )))
                dlg_self.class_names = ["Todas"] + unique_classes
                dlg_self.historial = historial

                dlg_self.combo = QComboBox()
                dlg_self.combo.addItems(dlg_self.class_names)
                dlg_self.combo.currentIndexChanged.connect(dlg_self.actualizar_tabla)

                dlg_self.table = QTableWidget()
                dlg_self.table.setColumnCount(5)
                dlg_self.table.setHorizontalHeaderLabels(
                    ["Imagen", "Predicción", "Confianza (%)", "Corregido", "Clase correcta"]
                )
                dlg_self.table.setAlternatingRowColors(True)
                dlg_self.thumbnail_size = 60

                layout = QVBoxLayout()
                layout.setContentsMargins(15, 15, 15, 15)
                layout.setSpacing(10)
                layout.addWidget(QLabel("Filtrar por clase:"))
                layout.addWidget(dlg_self.combo)
                layout.addWidget(dlg_self.table)
                dlg_self.setLayout(layout)

                dlg_self.actualizar_tabla()

            def actualizar_tabla(dlg_self):
                clase_filtrada = dlg_self.combo.currentText()
                if clase_filtrada == "Todas":
                    data = dlg_self.historial
                else:
                    data = [
                        h for h in dlg_self.historial
                        if h[1] == clase_filtrada or h[4] == clase_filtrada
                    ]
                dlg_self.table.setRowCount(len(data))
                for i, (ruta, prediccion, confianza, corregido, clase_correcta) in enumerate(data):
                    label_img = QLabel()
                    try:
                        img = Image.open(ruta).convert('RGB').resize(
                            (dlg_self.thumbnail_size, dlg_self.thumbnail_size)
                        )
                        arr_img = np.array(img)
                        h, w, ch = arr_img.shape
                        bytes_per_line = ch * w
                        qimg = QImage(arr_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        label_img.setPixmap(pixmap)
                    except Exception:
                        label_img.setText("No Img")
                    dlg_self.table.setCellWidget(i, 0, label_img)
                    dlg_self.table.setItem(i, 1, QTableWidgetItem(nombre_amigable(str(prediccion))))
                    dlg_self.table.setItem(i, 2, QTableWidgetItem("{:.2f}".format(confianza)))
                    dlg_self.table.setItem(i, 3, QTableWidgetItem(corregido))
                    dlg_self.table.setItem(i, 4, QTableWidgetItem(nombre_amigable(clase_correcta) if clase_correcta else ""))
                dlg_self.table.resizeColumnsToContents()
                dlg_self.table.setColumnWidth(0, dlg_self.thumbnail_size + 10)

        dlg = HistorialDialog(self.historial)
        dlg.exec_()

    # --------------------------------------------------------------
    def exportar_csv(self):
        if not self.historial:
            self.toast("No hay datos en el historial")
            return

        archivo, _ = QFileDialog.getSaveFileName(
            self, "Guardar historial como CSV", "historial.csv", "CSV files (*.csv)"
        )
        if not archivo:
            return

        try:
            with open(archivo, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Imagen", "Predicción", "Confianza", "Corregido", "Clase correcta"])
                for fila in self.historial:
                    nombre_archivo = os.path.basename(fila[0])
                    writer.writerow((nombre_archivo, *fila[1:]))
            self.toast(f"CSV exportado: {os.path.basename(archivo)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar el CSV:\n{e}")

    # --------------------------------------------------------------
    def exportar_pdf(self):
        if not self.historial:
            self.toast("No hay datos para el reporte PDF")
            return

        archivo, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte PDF", "reporte_comidas.pdf", "PDF files (*.pdf)"
        )
        if not archivo:
            return

        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(26, 58, 107)
            pdf.cell(0, 12, "Reporte - Clasificador de Comidas Peruanas",
                     ln=True, align="C")
            pdf.ln(4)

            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 6,
                     f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                     ln=True, align="C")
            pdf.ln(6)

            total = len(self.historial)
            aciertos = sum(1 for h in self.historial if h[3] == "Sí")
            correcciones = sum(1 for h in self.historial if h[3] == "No" and h[4] not in ("", "Ninguna"))
            errores = sum(1 for h in self.historial if h[4] == "Ninguna")
            porcentaje = (aciertos / total * 100) if total > 0 else 0

            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(26, 58, 107)
            pdf.cell(0, 8, "Resumen de la sesion", ln=True)
            pdf.ln(2)

            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 6, f"Total de imagenes procesadas: {total}", ln=True)
            pdf.cell(0, 6, f"Aciertos confirmados: {aciertos}", ln=True)
            pdf.cell(0, 6, f"Correcciones: {correcciones}", ln=True)
            pdf.cell(0, 6, f"Errores (ninguna clase): {errores}", ln=True)
            pdf.cell(0, 6, f"Porcentaje de acierto: {porcentaje:.2f}%", ln=True)
            pdf.ln(6)

            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(26, 58, 107)
            pdf.cell(0, 8, "Detalle de predicciones", ln=True)
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(26, 58, 107)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(70, 8, "Imagen", border=1, fill=True)
            pdf.cell(50, 8, "Prediccion", border=1, fill=True)
            pdf.cell(25, 8, "Conf. (%)", border=1, fill=True)
            pdf.cell(25, 8, "Corregido", border=1, fill=True)
            pdf.cell(0, 8, "Clase correcta", border=1, fill=True, ln=True)

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 50, 50)
            for fila in self.historial[:100]:
                ruta, pred, conf, corr, correcta = fila
                nombre = os.path.basename(ruta)
                if len(nombre) > 32:
                    nombre = nombre[:29] + "..."
                pred_txt = nombre_amigable(str(pred))
                if len(pred_txt) > 22:
                    pred_txt = pred_txt[:19] + "..."
                correcta_txt = nombre_amigable(correcta) if correcta else ""
                if len(correcta_txt) > 22:
                    correcta_txt = correcta_txt[:19] + "..."

                pdf.cell(70, 7, nombre, border=1)
                pdf.cell(50, 7, pred_txt, border=1)
                pdf.cell(25, 7, f"{conf:.1f}", border=1, align="C")
                pdf.cell(25, 7, corr if corr else "-", border=1, align="C")
                pdf.cell(0, 7, correcta_txt, border=1, ln=True)

            pdf.output(archivo)
            self.toast(f"PDF exportado: {os.path.basename(archivo)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo generar el PDF:\n{e}")

    # --------------------------------------------------------------
    def closeEvent(self, event):
        # >>> FIX: NO guardar historial, BORRARLO al salir
        self._borrar_historial_archivo()
        try:
            plt.close(self.fig)
        except Exception:
            pass
        event.accept()


# ------------------------------------------------------------------
# Punto de entrada
# ------------------------------------------------------------------
if __name__ == "__main__":
    if tflite is None:
        _abortar_sin_tflite()

    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    ventana = ClasificadorApp()
    ventana.show()
    sys.exit(app.exec_())