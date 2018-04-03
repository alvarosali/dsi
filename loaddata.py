# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:14:12 2016
"""
import csv, pydot
from sklearn.mixture import GaussianMixture
from sklearn import metrics, preprocessing, tree
import matplotlib.pyplot as plt
from sklearn.tree import _tree

def calc_metrics (data, labels_pred, labels_true):
    print "cálculo de métricas"
    # Homogeneity, completeness and V-measure¶
    # Given the knowledge of the ground truth class assignments of the samples, 
    # it is possible to define some intuitive metric using conditional entropy analysis.
    #   homogeneity: each cluster contains only members of a single class.
    #   completeness: all members of a given class are assigned to the same cluster.
    #   V-measure: armonic means of both
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_pred))
    
    # ADJUSTED RAND INDEX
    # Given the knowledge of the ground truth class assignments labels_true
    # and our clustering algorithm assignments of the same samples labels_pred, 
    # the adjusted Rand index is a function that measures the similarity of the two assignments, 
    # ignoring permutations and with chance normalization:
    # 1.0 is the perfect match score.
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    
    # the Mutual Information is a function that measures the agreement of the two assignments, ignoring permutations. 
    print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels))
    
    # The Silhouette Coefficient is defined for each sample and is composed of two scores:
    #    a: The mean distance between a sample and all other points in the same class.
    #    b: The mean distance between a sample and all other points in the next nearest cluster.
    # The Silhouette Coefficient s for a single sample is then given as: s = \frac{b - a}{max(a, b)}
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(data, labels))


"""
    funcion para cargar y tratar los datos
"""
def loaddata(filename, variables):
    print "carga y tratamiento de los datos"
    f = open(filename,'r')
    reader = csv.reader(f, delimiter=';')
    dat_num = []
    dat_infor = []
    for line in reader:
        # se dividen las variables de datos de las demás
        row1 = line[:7]
        row2 = []
        for j in variables:
            row2.append(line[j]) 
        if row2 != [] and 'n.d.' not in row2: 
            for i in range(0, len(row2)):
                # se tratan los datos para convertir strings a float
                row2[i] = row2[i].replace('.','')
                row2[i] = row2[i].replace(',','.')
            dat_num.append(list(map(float, row2)))
            dat_infor.append(row1+row2)

    f.close()
    return dat_infor, dat_num 


"""
    funcion para dibujar los datos
"""
def plotdata(data,labels,name): #def function plotdata
#colors = ['black']
    fig, ax = plt.subplots()
    plt.scatter([row[0] for row in data], [row[1] for row in data], c=labels)
    ax.grid(True)
    fig.tight_layout()
    plt.title(name)
    plt.show()

"""
    funcion para normalizar los valores introducidos
"""
def normalization(data):
	print("normalizacion de los datos")
	#http://scikit-learn.org/stable/modules/preprocessing.html
	min_max_scaler = preprocessing.MinMaxScaler()
	data = min_max_scaler.fit_transform(data)

	return data

"""
    funcion que devuelve el valor minimo de una lista
"""
def minimun_value(listValues): 
    minimun = listValues[0] 
    for value in listValues: 
        if value < minimun: 
            minimun = value 
    return minimun 

"""
    funcion que devuelve el maximo valor de una lista
"""
def maximun_value(listValues): 
    maximun = listValues[0] 
    for value in listValues: 
        if value > maximun: 
            maximun = value 
    return maximun 

"""
    funcion que devuelve el valor medio de una lista
"""
def average_value(listValues):
    average = 0
    for value in listValues: 
        average = average + value
    average = average / len(listValues)
    return average

"""
    funcion para calcular el minimo, el promedio y el maximo valor de una lista
"""
def calcValues(listValues):
    minimun = minimun_value(listValues)
    average = average_value(listValues)
    maximun = maximun_value(listValues)
    return minimun, average, maximun

"""
    funcion para realizar el algoritmo de expectation maximization
"""
def expectationmaximization(data):
    print "realizacion del algoritmo de expectation-maximization"
    em = GaussianMixture(n_components=7,covariance_type='full', init_params='kmeans')
    em.fit(data)
    labels =  em.predict(data)
    #plotdata(data,labels,'em')
    print ('EM results')

    return labels

"""
    funcion para crear los clusters y calcular sus valores minimos, medios y maximos
"""
def clustering(clusters, data, variables):
    print "creacion de los clusters y calculo de minimos, medias y maximos de los valores"
    file_write = open("solutions.txt", 'w')
    clusters_rows = []
    n_clusters = maximun_value(clusters)
    for i in range(n_clusters+1):
        clusters_rows.append([])

    i = 0
    for line in data:
        clusters_rows[clusters[i]].append(i)
        i = i + 1

    num_clus = 0
    # se calculan los minimos, maximos y medias de los valores de cada cluster
    for cluster_row in clusters_rows:

        values = []
        for cltr in range(len(variables)):
            values.append([])

        for cluster in cluster_row:
            splitted_cluster = []
            for variable in range(len(variables)):

                splitted_cluster.append(data[cluster][variable])

            i = 0
            for value in values:
                value.append(float(splitted_cluster[i]))
                i = i + 1
            
        file_write.write("\nminimos, medias y maximos del cluster numero: " + str(num_clus) + ' con ' + str(len(cluster_row)) + ' municipios' '\n\n' )

        file_write.write(str(cluster_row) + '\n\n')

        num_clus = num_clus + 1
        for value in values:
            file_write.write(str(calcValues(value)) + '\n')

    file_write.close()

    return clusters_rows

"""
    funcion para dibujar el arbol de decision correspondiente 
    al clustering. Se crea el archivo .dot y el .pdf
"""
def decisionTree(clusters, data, variables):
    X = data
    Y = clusters

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    
    all_names = ['cod_INE', 'com_aut', 'cod_INE_prov', 'prov', 'cod_INE_mun', 'cod_INE_mun_5', 'mun', 'n_obs', 'poblacion', 'poblacion_decl', 'IRPF', 'IRPF_decl', 'IRPF_hab', 'renta_media', 'i_gini', 'i_atkinson', 'conc_1', 'conc_0,5', 'conc_0,1']
    
    # se realiza el arbol de decision con los datos del clustering y con las variables selecionadas
    names = []
    for i in variables:
        names.append(all_names[i])

    with open('graph.dot', 'w') as f:
        f = tree.export_graphviz(clf, out_file=f,feature_names=names)

    graphs = pydot.graph_from_dot_file('graph.dot')
    print("arbol de decision en formato .dot creado correctamente")

    graphs[0].write_pdf('graph.pdf')
    print("arbol de decision en formato .pdf creado correctamente")


def porc_com_aut(data, clusters_rows):
    
    print "\n\nporcentaje de cluster por comunidad autonoma\n\n"

    print [' 1  ',' 2  ',' 3  ',' 4  ',' 5  ',' 6  ',' 7  ',' 8  ',' 9  ',' 10 ',' 11 ',' 12 ',' 13 ',' 14 ', ' 15 ',' 16 ',' 17 ']
    rows = []
    for dat in data:
        if dat[0] != '':
            rows.append(int(dat[0]))

    n_coms_aut= maximun_value(rows)

    tot_coms_aut = []

    """
        se calcula el porcentaje de cluster que aparece en cada comunidad autonoma
        filas: clusters
        columnas: comunidades autonomas
        * se descartan ceuta y melilla
        * no hay datos de los municipios de Navarra y Pais Vasco
    """

    for cluster_row in clusters_rows:

        coms_aut1 = []
        coms_aut2 = []
        
        porc_tot_coms = []
        for i in range(n_coms_aut):
            coms_aut1.append(0)
            coms_aut2.append(0)
            porc_tot_coms.append([])

        for cl in cluster_row:
            if data[cl][0] != '':
                coms_aut1[int(data[cl][0])-1] = coms_aut1[int(data[cl][0])-1] + 1

        tot_coms_aut.append(coms_aut1)

        i = 0
        for com_aut in coms_aut1:
            coms_aut2[i] = format((com_aut/float(len(cluster_row)))*100.0,'.2f')
            i = i + 1

        print coms_aut2

    """
        se calcula el porcentaje de municipios de una misma comunidad autonoma que aparecen en cada cluster
        filas: comunidades autonomas
        columnas: clusters
        *se descartan ceuta y melilla
        * no hay datos de los municipios de Navarra y Pais Vasco
    """
        
    for tot_coms in tot_coms_aut:

        cont = 0
        for com in tot_coms:

            porc_tot_coms[cont].append(com);
            cont = cont + 1

    for porc_tot_com in porc_tot_coms:
        tot = 0
        for i in porc_tot_com:
            tot = tot + i
        cont = 0
        for i in porc_tot_com:
            if tot != 0:
                porc_tot_com[cont] = format((i/float(tot))*100, '.2f')
            cont = cont +1

    print "\n\nporcentaje de municipios de una comunidad autonoma por cluster\n\n"
    for porc_tot_com in porc_tot_coms:
        print porc_tot_com


filename = 'datos.csv'



# 1. Se introducen los indices de las variables del fichero de datos que se desean incluir en el analisis
variables = [11, 12, 13, 15, 16]

# 2. se cargan los datos 
all_data, data = loaddata(filename, variables) 

labels = [0 for x in range(len(data))]
#plotdata(data,labels,'basic')

# 3. se normalizan los datos
data = normalization(data)

# 4. se realiza el analisis y clustering EM
labels = expectationmaximization(data)

labels_true = []
for line in all_data:
    labels_true.append(line[0])

# 5. se calculan las métricas
calc_metrics(data,labels, labels_true)

# 6. se crean y analizan los clusters obtenidos
clusters = clustering(labels, data, variables)

# 7. se dibuja el arbol de decision
decisionTree(labels, data, variables)

# 8. se dibujan las tablas con la informacion de los municipios y clusters obtenidos
porc_com_aut(all_data, clusters)


















