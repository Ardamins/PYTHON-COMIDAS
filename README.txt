=========================================================
 CLASIFICADOR DE COMIDAS PERUANAS - README
=========================================================

Este software permite clasificar imágenes de platos peruanos utilizando un modelo de Inteligencia Artificial (IA) y una interfaz gráfica amigable desarrollada con PyQt5.

---------------------------------------------------------
REQUISITOS
---------------------------------------------------------
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

---------------------------------------------------------
DEPENDENCIAS DE PYTHON
---------------------------------------------------------
Antes de ejecutar el programa, instala las siguientes librerías desde una terminal/cmd:

pip install pyqt5 pillow numpy matplotlib tensorflow

---------------------------------------------------------
ESTRUCTURA DE CARPETAS Y ARCHIVOS
---------------------------------------------------------

Tu carpeta debe verse así:

/codigo_fuente/
│
├── app.py
├── app.spec
├── modelo_comidas_peruanas.tflite
├── comidas/
│    ├── lomo_saltado/
│    ├── papa_rellena_peruana/
│    └── ... (otras clases de comida)
├── imagenes_corregidas/
│    └── (puede estar vacía o con imágenes corregidas)
└── README.txt

---------------------------------------------------------
¿CÓMO EJECUTAR EL PROGRAMA?
---------------------------------------------------------

1. Abre una terminal (cmd) y navega a la carpeta donde tienes el archivo `app.py`:

   cd ruta/a/tu/codigo_fuente

2. Ejecuta el programa con:

   python app.py

# Clonar el repositorio
git clone https://github.com/Ardamins/PYTHON-COMIDAS.git
cd PYTHON-COMIDAS

# Crear el venv con Python 3.10
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4
python -m pip install tensorflow-cpu==2.16.2
python -m pip install PyQt5 matplotlib pillow

# Ejecutar
python app.py