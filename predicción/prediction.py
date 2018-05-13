# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

"""
    funcion para modificar el formato de los numeros mal redactados, se introduce el
    numero a modificar (number) y la cantidad de cifras enteras que contiene (n_int).
"""
def formatN(number, n_int):
	if type(number) is str:
		number = number.replace('.','')

		number = float(number[:n_int] + '.' + number[n_int:])
		number = round(number,2)
	
	return number

"""
    funcion para cargar los casos de dengue por semanas
"""
def loadcases(filename):
    print "carga y tratamiento de los casos de dengue por semanas"
    f = open(filename,'rU')
    reader = csv.reader(f, delimiter=';')
    data_cases = []
    for line in reader:
    	if line[1] != 'year':
    		data_cases.append([line[0],int(line[1]), int(line[2]), int(line[3])])
        

    f.close()
    
    return data_cases

"""
    funcion para cargar y formatear las caracteristicas meteorologicas de las semanas
"""
def loadfeatures(filename):
	print "carga y formato de las caracteristicas meteorologicas de las semanas"
	f = open(filename,'rU')
	reader = csv.reader(f, delimiter=';')
	data_features = []
	line_prev = []
	for line in reader:
		line.pop(3)
		line.pop(3)
		line.pop(3)
		line.pop(3)
		line.pop(3)
		if line[1] != 'year':
			if '' in line:
				for index, y in enumerate(line):
					if y == '':
						line[index] = line_prev[index]

			line[1] = int(line[1])
			line[2] = int(line[2])
			line[3] = float(line[3])
			line[4] = formatN(line[4],3)
			line[5] = formatN(line[5],3)
			line[6] = formatN(line[6],3)
			line[7] = float(line[7])
			line[8] = float(line[8])
			line[9] = float(line[9])
			line[10] = formatN(line[10],2)
			line[11] = float(line[11])
			line[12] = formatN(line[12],2)
			line[13] = formatN(line[13],1)
			line[14] = formatN(line[14],2)
			line[15] = formatN(line[15],1)
			line[16] = float(line[16])
			line[17] = float(line[17])
			line[18] = float(line[18])

			data_features.append(line)
			line_prev = line

	f.close()

	return data_features

"""
    funcion para separar los datos de cada una se las ciudades 
    y asi obtener una lista con la informacion de cada ciudad
"""
def citiesSeparate(array):
	name_cities = []
	name_cities.append(array[0][0])
	cont = 0
	info_cities = []
	info_cities.append([])

	for data in array:
		if data[0] == name_cities[cont]:
			data.pop(0)
			data.pop(0)
			data.pop(0)
			info_cities[cont].append(data)
		else:
			name_cities.append(data[0])
			info_cities.append([])
			cont += 1
			data.pop(0)
			data.pop(0)
			data.pop(0)
			info_cities[cont].append(data)

	return name_cities, info_cities

"""
    funcion para realizar el proceso de arbol de decision (random forest) 
"""
def decisionTree(X, y, X_t):

	#1.1 Model Parametrization 
	regressor = RandomForestRegressor(n_estimators= 20, max_depth = 4, criterion='mae', random_state=None)
	y_prediction = []
	for i in range(len(X)):
		# sample a training set while holding out 40% of the data for testing (evaluating) our classifier:
		X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], test_size=0.35)

		#1.2 Model construction
		regressor.fit(X_train, y_train)

		# Test
		y_pred = regressor.predict(X_test)
		#print y_pred

		y_pred2 = regressor.predict(X_t[i])

		y_prediction.append(y_pred2)

		# metrics calculation 
		mae = mean_absolute_error(y_test,y_pred)
		print "Error Measure ", mae

		#plt.subplot(2, 1, i + 1)
		# x axis for plotting
		
		xx = np.stack(i for i in range(len(y_test)))
		plt.scatter(xx, y_test, c='r', label='data train')
		plt.plot(xx, y_pred, c='g', label='prediction train')
		xx2 = np.stack(i for i in range(len(y_pred2)))
		plt.scatter(xx2, y_pred2, c='b', label='data predict')
		plt.axis('tight')
		plt.legend()
		plt.title("Random Forests")

		plt.show()
		

	return y_prediction
	    

"""
    funcion para incluir la solucion en el fichero de Submission_Format
"""
def printSol(data_cases, sol):
	i = 0
	j = 0
	with open('solution.csv', 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',')
		spamwriter.writerow(['city', 'year', 'weekofyear', 'total_cases'])
		for line in data_cases:
			if i < len(sol):
				if j < len(sol[i]):
					line[3] = int(sol[i][j])
					spamwriter.writerow(line)
					j+=1
				else:
					i+=1
					j=0
					line[3] = int(sol[i][j])
					spamwriter.writerow(line)







# ficheros de datos
filenameCases = 'Training_Data_Labels.csv'
filenameFeatures = 'Training_Data_Features.csv'
filenameCasesTest = 'Submission_Format.csv'
filenameFeaturesTest = 'Test_Data_Features.csv'

# 1. se cargan los casos de dengue por semanas
data_cases = loadcases(filenameCases)
data_cases_test = loadcases(filenameCasesTest)

# 2. se cargan y se formatean las caracteristicas meteorologicas de las semanas
data_features = loadfeatures(filenameFeatures)
data_features_test = loadfeatures(filenameFeaturesTest)

# 3. se crean las tarjetas de datos por ciudades para el entrenamiento del arbol de decision
name_cities_cases, info_cities_cases = citiesSeparate(data_cases)
name_cities_features, info_cities_features = citiesSeparate(data_features)
name_cities_features_test, info_cities_features_test = citiesSeparate(data_features_test)


if name_cities_features != name_cities_cases:
	print 'error! los ficheros de casos y caracteristicas meteorilogicas no coinciden'


# 4. Se ejecuta el algoritmo de arbol de decision (random forest) y se obtiene una solucion 
sol = decisionTree(info_cities_features, info_cities_cases, info_cities_features_test)

# 5. Se complementa el Submission_Format con las solucion obtenida
e = printSol(data_cases_test, sol)







    


















