# -*- coding: utf-8 -*-
import sys
import os
import shutil
import csv
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QMessageBox, QInputDialog, QTableWidget, QTableWidgetItem, QDialog,
    QProgressBar, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QStandardPaths
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter


# ------------------------------------------------------------------
# Importación de TFLite con triple fallback:
#   1) tflite_runtime  (más ligero, no carga TensorFlow completo)
#   2) tensorflow.lite.python.interpreter  (sin cargar pywrap completo)
#   3) tensorflow.lite  (último recurso)
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
# Rutas adaptadas para PyInstaller (onefile / onedir)
# ------------------------------------------------------------------
def resource_path(relative_path):
    """Ruta para recursos empaquetados (solo lectura)."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def writable_path(relative_path=""):
    """Ruta para escritura: carpeta de datos de usuario o junto al .exe."""
    if getattr(sys, 'frozen', False):
        # En modo empaquetado, escribir junto al exe puede fallar por permisos.
        # Usamos la carpeta de datos de la aplicación del usuario.
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not base:
            base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path) if relative_path else base


BASE_DIR = writable_path()
MODEL_PATH = resource_path("modelo_comidas_peruanas.tflite")
CLASS_DIR = resource_path("comidas")

IMG_SIZE = (224, 224)

# Lista de clases a partir de las subcarpetas de "comidas"
if os.path.isdir(CLASS_DIR):
    CLASS_NAMES = sorted(
        d for d in os.listdir(CLASS_DIR)
        if os.path.isdir(os.path.join(CLASS_DIR, d))
    )
else:
    CLASS_NAMES = []


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
            "Coloca 'modelo_comidas_peruanas.tflite' junto al script o "
            "inclúyelo con --add-data al empaquetar."
        )
        sys.exit(1)

    if _USE_TFLITE_RUNTIME:
        interp = tflite.Interpreter(model_path=MODEL_PATH)
    else:
        interp = tflite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp


interpreter = _cargar_interprete()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predecir_imagen(image_path):
    """Devuelve (clase, confianza_en_porcentaje)."""
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)

    if input_details[0]['dtype'] == np.uint8:
        array = np.array(img).astype(np.uint8)
    else:
        array = np.array(img).astype(np.float32) / 255.0
        array = array.astype(input_details[0]['dtype'])

    array = np.expand_dims(array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_index = int(np.argmax(output))
    if pred_index >= len(CLASS_NAMES):
        return "Desconocido", 0.0
    clase = CLASS_NAMES[pred_index]
    confianza = float(np.max(output)) * 100
    return clase, confianza


# ------------------------------------------------------------------
# Ventana principal
# ------------------------------------------------------------------
class ClasificadorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificador de Comidas Peruanas")
        self.resize(420, 780)

        # --- Widgets ---
        self.label_imagen = QLabel("Selecciona una imagen para predecir", self)
        self.label_imagen.setFixedSize(224, 224)
        self.label_imagen.setStyleSheet("border: 1px solid #ccc; background: #fafafa;")
        self.label_imagen.setAlignment(Qt.AlignCenter)

        self.btn_seleccionar = QPushButton("Seleccionar Imagen", self)
        self.btn_seleccionar.clicked.connect(self.seleccionar_imagen)

        self.label_resultado = QLabel("Predicción: ", self)
        self.label_resultado.setStyleSheet("font-size: 16px; margin-top: 20px;")
        self.label_resultado.setWordWrap(True)

        self.btn_feedback = QPushButton("¿La predicción es correcta?", self)
        self.btn_feedback.clicked.connect(self.dar_feedback)
        self.btn_feedback.setEnabled(False)

        self.btn_historial = QPushButton("Ver historial de predicciones", self)
        self.btn_historial.clicked.connect(self.mostrar_historial)

        self.btn_exportar = QPushButton("Exportar historial a CSV", self)
        self.btn_exportar.clicked.connect(self.exportar_csv)

        self.label_metricas = QLabel("", self)
        self.label_metricas.setStyleSheet("font-size: 13px; color: #444; margin-top: 15px;")
        self.label_metricas.setWordWrap(True)

        self.progress = QProgressBar(self)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setFormat("Porcentaje de acierto: %p%")

        self.fig, self.ax = plt.subplots(figsize=(4, 2))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedHeight(200)

        # --- Layout ---
        layout = QVBoxLayout()
        layout.addWidget(self.label_imagen, alignment=Qt.AlignCenter)
        layout.addWidget(self.btn_seleccionar)
        layout.addWidget(self.label_resultado)
        layout.addWidget(self.label_metricas)
        layout.addWidget(self.progress)
        layout.addWidget(self.canvas)
        layout.addWidget(self.btn_feedback)
        layout.addWidget(self.btn_historial)
        layout.addWidget(self.btn_exportar)
        self.setLayout(layout)

        # --- Estilo global (QSS) ---
        self.setStyleSheet("""
            QWidget {
                background: #f6f7fb;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                color: #333;
            }
            QLabel { font-size: 15px; }
            QPushButton {
                background-color: #16447f;
                color: #fff;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 15px;
                margin: 4px 0;
            }
            QPushButton:hover { background-color: #2163b3; }
            QPushButton:disabled { background-color: #9aa7bd; }
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 8px;
                background: #fff;
                height: 20px;
                margin: 8px 0;
            }
            QProgressBar::chunk {
                background-color: #2163b3;
                border-radius: 8px;
            }
            QTableWidget {
                border: 1px solid #bbb;
                border-radius: 8px;
                background: #fff;
                font-size: 13px;
            }
            QHeaderView::section {
                background: #16447f;
                color: #fff;
                font-weight: bold;
                font-size: 14px;
                border: none;
                padding: 3px;
            }
            QComboBox {
                border: 1px solid #bbb;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 15px;
                background: #fff;
            }
        """)

        # --- Variables de estado ---
        self.ultima_imagen_path = None
        self.ultima_prediccion = None
        self.historial = []  # (ruta, pred, conf, corregido, clase_correcta)
        self.class_names = list(CLASS_NAMES)

    # --------------------------------------------------------------
    def seleccionar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if not ruta:
            self.label_resultado.setText("No se seleccionó ninguna imagen.")
            self.btn_feedback.setEnabled(False)
            return

        # Mostrar miniatura
        img = Image.open(ruta).convert('RGB').resize(IMG_SIZE)
        arr_img = np.array(img)
        h, w, ch = arr_img.shape
        bytes_per_line = ch * w
        qimg = QImage(arr_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label_imagen.setPixmap(pixmap)

        try:
            clase, confianza = predecir_imagen(ruta)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo predecir la imagen:\n{e}")
            return

        if confianza <= 35.0:
            self.label_resultado.setText(
                "No se reconoce ninguna clase con suficiente confianza.\n"
                f"Confianza: {confianza:.2f}%"
            )
            self.ultima_imagen_path = ruta
            self.ultima_prediccion = None
            self.btn_feedback.setEnabled(False)
            self.historial.append((ruta, "Ninguna", confianza, "", ""))
            self.actualizar_metricas()
            self.agregar_nueva_clase(ruta)
        else:
            self.label_resultado.setText(
                f"Predicción: {clase}\nConfianza: {confianza:.2f}%"
            )
            self.ultima_imagen_path = ruta
            self.ultima_prediccion = clase
            self.btn_feedback.setEnabled(True)
            self.historial.append((ruta, clase, confianza, "", ""))
            self.actualizar_metricas()

    # --------------------------------------------------------------
    def _guardar_imagen_en_clase(self, ruta_imagen, nueva_clase):
        """Copia la imagen a imagenes_corregidas/<nueva_clase>/ y devuelve la ruta."""
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
            "¿Deseas agregar una nueva clase con esta imagen?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        nueva_clase, ok = QInputDialog.getText(
            self, "Nueva clase", "Escribe el nombre de la nueva clase:"
        )
        if ok and nueva_clase.strip():
            self._guardar_imagen_en_clase(ruta_imagen, nueva_clase.strip())
            QMessageBox.information(
                self, "¡Guardado!",
                f"Imagen guardada como ejemplo de la nueva clase '{nueva_clase}'."
            )
        else:
            QMessageBox.information(self, "Cancelado", "No se agregó ninguna clase nueva.")

    # --------------------------------------------------------------
    def dar_feedback(self):
        if self.ultima_imagen_path is None:
            return

        resp = QMessageBox.question(
            self, "Confirmar",
            f"¿La predicción ({self.ultima_prediccion}) es correcta?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        index = len(self.historial) - 1

        if resp == QMessageBox.Yes:
            QMessageBox.information(self, "¡Gracias!", "¡Gracias por tu confirmación!")
            if index >= 0:
                ruta, prediccion, confianza, _, _ = self.historial[index]
                self.historial[index] = (ruta, prediccion, confianza, "Sí", prediccion)
            self.actualizar_metricas()

        elif resp == QMessageBox.No:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("¿Qué quieres hacer?")
            dialog.setText("¿Qué deseas hacer con esta imagen?")
            btn_agregar = dialog.addButton("Agregar clase nueva", QMessageBox.ActionRole)
            btn_ninguna = dialog.addButton("No es ninguna", QMessageBox.ActionRole)
            dialog.addButton(QMessageBox.Cancel)
            dialog.exec_()
            selected = dialog.clickedButton()

            if selected == btn_agregar:
                nueva_clase, ok = QInputDialog.getText(
                    self, "Nueva clase", "Escribe el nombre de la nueva clase:"
                )
                if ok and nueva_clase.strip():
                    self._guardar_imagen_en_clase(
                        self.ultima_imagen_path, nueva_clase.strip()
                    )
                    QMessageBox.information(
                        self, "¡Guardado!",
                        f"Imagen guardada como ejemplo de la nueva clase '{nueva_clase}'."
                    )
                    if index >= 0:
                        ruta, prediccion, confianza, _, _ = self.historial[index]
                        self.historial[index] = (ruta, prediccion, confianza, "No", nueva_clase.strip())
                    self.actualizar_metricas()

            elif selected == btn_ninguna:
                if index >= 0:
                    ruta, prediccion, confianza, _, _ = self.historial[index]
                    self.historial[index] = (ruta, prediccion, confianza, "No", "Ninguna")
                QMessageBox.information(self, "Registrado", "Marcado como ninguna clase.")
                self.actualizar_metricas()

    # --------------------------------------------------------------
    def actualizar_metricas(self):
        total = len(self.historial)
        aciertos = sum(1 for h in self.historial if h[3] == "Sí")
        correcciones = sum(1 for h in self.historial if h[3] == "No" and h[4] not in ("", "Ninguna"))
        errores_ninguna = sum(1 for h in self.historial if h[4] == "Ninguna")
        porcentaje = (aciertos / total * 100) if total > 0 else 0

        self.label_metricas.setText(
            f"Imágenes procesadas: {total} | "
            f"Aciertos: {aciertos} | "
            f"Correcciones: {correcciones} | "
            f"Errores (ninguna): {errores_ninguna} | "
            f"Porcentaje de acierto: {porcentaje:.2f}%"
        )
        self.progress.setValue(int(porcentaje))

        clases_predichas = []
        for h in self.historial:
            if h[4] == "Ninguna":
                clases_predichas.append("Ninguna")
            elif h[3] == "Sí" or (h[3] == "No" and h[4]):
                clases_predichas.append(h[4])

        conteos = Counter(clases_predichas)
        clases = list(conteos.keys())
        valores = list(conteos.values())
        colores = ["orangered" if c == "Ninguna" else "skyblue" for c in clases]

        self.ax.clear()
        if clases:
            self.ax.bar(clases, valores, color=colores)
            self.ax.set_ylabel("Cantidad")
            self.ax.set_xlabel("Clase")
            self.ax.set_title("Predicciones por clase (errores en rojo)")
            self.ax.tick_params(axis='x', rotation=30)
        else:
            self.ax.text(0.5, 0.5, 'Aún sin datos', ha='center', va='center')
        self.fig.tight_layout()
        self.canvas.draw()

    # --------------------------------------------------------------
    def mostrar_historial(self):
        class HistorialDialog(QDialog):
            def __init__(dlg_self, historial):
                super().__init__()
                dlg_self.setWindowTitle("Historial de predicciones")

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
                dlg_self.thumbnail_size = 64

                layout = QVBoxLayout()
                layout.addWidget(QLabel("Filtrar por clase:"))
                layout.addWidget(dlg_self.combo)
                layout.addWidget(dlg_self.table)
                dlg_self.setLayout(layout)
                dlg_self.resize(780, 400)

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
                    dlg_self.table.setItem(i, 1, QTableWidgetItem(str(prediccion)))
                    dlg_self.table.setItem(i, 2, QTableWidgetItem("{:.2f}".format(confianza)))
                    dlg_self.table.setItem(i, 3, QTableWidgetItem(corregido))
                    dlg_self.table.setItem(i, 4, QTableWidgetItem(clase_correcta))
                dlg_self.table.resizeColumnsToContents()
                dlg_self.table.setColumnWidth(0, dlg_self.thumbnail_size + 10)

        dlg = HistorialDialog(self.historial)
        dlg.exec_()

    # --------------------------------------------------------------
    def exportar_csv(self):
        if not self.historial:
            QMessageBox.information(self, "Exportar", "No hay datos en el historial.")
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
            QMessageBox.information(self, "Exportar", f"Historial exportado como:\n{archivo}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar el CSV:\n{e}")

    # --------------------------------------------------------------
    def closeEvent(self, event):
        """Cierra la figura de matplotlib al salir para liberar memoria."""
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
    ventana = ClasificadorApp()
    ventana.show()
    sys.exit(app.exec_())