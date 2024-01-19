## Detección de emociones en tiempo real ##
## SISTEMAS INTELIGENTES ##
## https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg
## David Revelo Luna

# Import de librerias
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import time
import datetime
import pandas as pd

#inicio
inicio = datetime.datetime.now()

# Variables para calcular FPS
time_actualframe = 0
time_prevframe = 0

# Tipos de emociones del detector
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Cargamos el  modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
emotionModel = load_model("modelFEC.h5")

# Crear DataFrame vacío para almacenar los datos
df = pd.DataFrame(columns=[
    'x.day', 'x.month', 'x.year', 'porcent_emo', 'emo'
	])

# Se crea la captura de video
#Cámara Interna del Pc
#cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#Cámara externa
cam = cv2.VideoCapture(1,cv2.CAP_DSHOW)

# Toma la imagen, los modelos de detección de rostros y mascarillas 
# Retorna las localizaciones de los rostros y las predicciones de emociones de cada rostro
def predict_emotion(frame,faceNet,emotionModel):
	# Construye un blob de la imagen
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

	# Realiza las detecciones de rostros a partir de la imagen
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Listas para guardar rostros, ubicaciones y predicciones
	faces = []
	locs = []
	preds = []
	
	# Recorre cada una de las detecciones
	for i in range(0, detections.shape[2]):
		
		# Fija un umbral para determinar que la detección es confiable
		# Tomando la probabilidad asociada en la deteccion

		if detections[0, 0, i, 2] > 0.4:
			# Toma el bounding box de la detección escalado
			# de acuerdo a las dimensiones de la imagen
			box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
			(Xi, Yi, Xf, Yf) = box.astype("int")

			# Valida las dimensiones del bounding box
			if Xi < 0: Xi = 0
			if Yi < 0: Yi = 0
			
			# Se extrae el rostro y se convierte BGR a GRAY
			# Finalmente se escala a 224x244
			face = frame[Yi:Yf, Xi:Xf]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			face = cv2.resize(face, (48, 48))
			face2 = img_to_array(face)
			face2 = np.expand_dims(face2,axis=0)
			# Se agrega los rostros y las localizaciones a las listas
			faces.append(face2)
			locs.append((Xi, Yi, Xf, Yf))
			pred = emotionModel.predict(face2)
			preds.append(pred[0])

	return (locs,preds)

while True:
	# Se toma un frame de la cámara y se redimensiona
	ret, frame = cam.read()
	frame = imutils.resize(frame, width=640)

	(locs, preds) = predict_emotion(frame,faceNet,emotionModel)
	
	# Para cada hallazgo se dibuja en la imagen el bounding box y la clase
	
	for (box, pred) in zip(locs, preds):
		x = datetime.datetime.now()
		print(x,x.strftime("%A"))

		(Xi, Yi, Xf, Yf) = box
		(angry,disgust,fear,happy,neutral,sad,surprise) = pred
		label = ''
		# Se agrega la probabilidad en el label de la imagen
		label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(angry,disgust,fear,happy,neutral,sad,surprise) * 100)
		emo=format(classes[np.argmax(pred)])
		print(label, "time_actualframexx:",)
		cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255,0,0), -1)
		cv2.putText(frame, label, (Xi+5, Yi-15),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
		cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255,0,0), 3)
		# Formato personalizado de fecha
		formatted_date = "{}/{}/{}: {} Hora: {}, Minuto: {}, Segundos: {}, Milisegundo: {}".format(
    	x.day, x.month, x.year, x.strftime("%A"), x.strftime("%H"), x.strftime("%M"), x.strftime("%S"), x.strftime("%f")[:3])
		porcent_emo=max(angry,disgust,fear,happy,neutral,sad,surprise) * 100
		print("Porcentaje de la emoción %: ", porcent_emo)
		print("Emoción exclusiva:", emo)
		print("@Emoción y porcentaje identificado por el algoritmo:",label)	
	# Imprimir el resultado
		print("FECHA DE LA EMOCIÓN:", formatted_date)

	#Noadd
# Crear DataFrame
		data = {
    	'x.day': [x.day],
    	'x.month': [x.month],
    	'x.year': [x.year],
    	'x.strftime("%A")': [x.strftime("%A")],
    	'x.strftime("%H")': [x.hour],
    	'x.strftime("%M")': [x.minute],
    	'x.strftime("%S")': [x.second],
    	'x.strftime("%f")[:3]': [x.strftime("%f")[:3]],
    	'porcent_emo': [porcent_emo],
    	'emo': [emo],
		'horact':[int((x - inicio).total_seconds())]
	}
	#df = df.append(pd.DataFrame(data), ignore_index=True)
		df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
		
	time_actualframe = time.time()
	if time_actualframe>time_prevframe:
		fps = 1/(time_actualframe-time_prevframe)
	
	time_prevframe = time_actualframe

	cv2.putText(frame, str(int(fps))+" FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		 break
print('Fin')

# Imprimir el DataFrame
print(df)
# Guardar el DataFrame en un archivo CSV

dfHora=	df['horact'].copy()
dfEmo=df['porcent_emo'].copy()
plt.scatter(dfHora,dfEmo)
plt.title('Intensidad de la emoción en la evaluación')
plt.xlabel('Segundos de la evaluación')
plt.ylabel('Porcentaje de emoción')
plt.show()
"""""
'angry','disgust','fear','happy','neutral','sad','surprise'
"""""
df[df['emo']=="sad"]['horact'].hist(alpha=0.9, color='black', label='Sad' )
df[df['emo']=="happy"]['horact'].hist(alpha=0.7, color='green', label='Happy')
df[df['emo']=="neutral"]['horact'].hist(alpha=0.7, color='blue', label='Neutral')
df[df['emo']=="angry"]['horact'].hist(alpha=0.7, color='red', label='Angry')
df[df['emo']=="fear"]['horact'].hist(alpha=0.7, color='yellow', label='Fear')
df[df['emo']=="surprise"]['horact'].hist(alpha=0.7, color='orange', label='Surprise')

plt.legend(["Sad", "Happy","Neutral", "Angry", "Fear", "Surprise" ])
plt.title("Análsis de la cualidad de las emociones")
plt.xlabel('Segundos de la evaluación')
plt.ylabel('Nº de veces que se ha presentado la emoción')
plt.show()

# Imprimir mensaje de confirmación
print("DataFrame guardado como resultadoEva02.csv")

cv2.destroyAllWindows()
cam.release()