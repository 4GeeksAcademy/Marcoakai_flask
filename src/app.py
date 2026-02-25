from utils import db_connect
engine = db_connect()

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# cargar pipeline completo
with open('models/modelo.pkl', 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)

@app.route('/', methods=['GET', 'POST'])
def inicio():
    resultado_final = None
    
    if request.method == 'POST':
        try:
            duracion = float(request.form['duracion'])
            votos = float(request.form['votos'])
            puntuacion_meta = float(request.form['puntuacion_meta'])
            
            datos_ingresados = np.array([[duracion, votos, puntuacion_meta]])
            
            # el pipeline ya tiene imputador dentro
            prediccion_cruda = modelo_cargado.predict(datos_ingresados)[0]
            resultado_final = round(prediccion_cruda, 2)
            
        except:
            resultado_final = "Error en los datos"

    return render_template('index.html', prediccion=resultado_final)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

