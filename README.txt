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

---------------------------------------------------------
DESCRIPCIÓN BREVE
---------------------------------------------------------
- El sistema permite seleccionar imágenes y clasificar automáticamente el plato peruano.
- Puedes dar retroalimentación y corregir la clase, además de agregar nuevas clases.
- Guarda un historial de predicciones, muestra métricas y permite exportar el historial a CSV.
- Las imágenes corregidas se guardan en la carpeta `imagenes_corregidas/` para posibles futuros reentrenamientos.

---------------------------------------------------------
CONTACTO Y AUTORES
---------------------------------------------------------
- [Luis Anderson Xavier Condor Miranda]
- [acondormi@ucvvirtual.edu.pe]

---------------------------------------------------------
NOTAS
---------------------------------------------------------
- Si tienes errores con librerías, asegúrate de tener Python y pip correctamente instalados.
- Si usas Windows, puedes crear un entorno virtual para evitar conflictos de dependencias.
- Para volver a generar el ejecutable, utiliza el archivo `app.spec` y PyInstaller.

=========================================================
