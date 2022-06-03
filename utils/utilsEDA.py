# Librerias 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import plotly.express as px
import scipy.stats as ss
import requests
import json
import time
import os 
import statsmodels.api as sm

# ----------
# FUNCIONES
# ----------

            ###
# **Funciones limpieza:**
            ###

def comp_colna_filna(df_pcp,col=-1):
    '''
    Objetivo: realizar una comprobación las columnas y filas del df para ver si tienen alguna completamente llena de NaN.
                En este caso simplemente nos lo informa pero no realiza ningún cambio en el dataset original.

    args.
    -----
    df_pcp: pd.DataFrame; es el dataset/dataframe principal, que vamos a comparar con el que evalúa si hay nan en filas y columnas.
    col: lista de str; lista de las columnas que quieras eliminar previamente.

    ret.
    ----
    No hay return, simplemente printea la respuesta.
    '''
    # Comprobamos si existe alguna fila o columna completamente llena de NaN
    df=df_pcp.copy()
    if col!=-1:
        df_aux=df.loc[:,col]
        df=df.drop(columns=col)
    df=df.dropna(how='all')
    df=df.dropna(axis=1,how='all')
    df=df_aux.join(df)
    # Se comprueba si no hay ninguna columna ni fila que se elimine
    print('Se mantienen las mismas filas:',df.shape[0]==df_pcp.shape[0],'y columnas:',df.shape[1]+len(col)==df_pcp.shape[1])
    print('Número fil y col antes',df_pcp.shape,'Número fil y col después',df.shape)

def del_colna_filna(df,col=-1):
    '''
    Objetivo: realizar una comprobación las columnas y filas del df para ver si tienen alguna completamente llena de NaN y en tal caso las elimina. 

    args.
    -----
    df: pd.DataFrame; es el dataset/dataframe principal, que vamos a comparar con el que evalúa si hay nan en filas y columnas.
    col: lista de str; lista de las columnas que quieras eliminar previamente. (opcional)

    ret.
    ----
    df: dataset/dataframe actualizado.
    '''
    # Comprobamos si existe alguna fila o columna completamente llena de NaN y en tal caso la eliminamos del df
    if col!=-1:
        df_aux=df.loc[:,col]
        df=df.drop(columns=col)
    df.dropna(how='all',inplace=True)
    df.dropna(axis=1,how='all',inplace=True)
    return df_aux.join(df)

                ###
# **Funciones acomodar dataset:**
                ###

def transf_dt(df,colstr,change='%Y-%m-%d %H:%M:%S'):
    '''
    Objetivo: transformar unna columna de tipo str en tipo dtime, haciendo una copia de la original.

    args.
    -----
    df: pd.DataFrame; es el dataset que contiene la columna que se quiere copiar.
    colstr: str; nombre columna que se quiere pasar a dt
    change: str; 

    ret.
    ----
    x: columna transformada en dtime
    
    '''

    x=[]
    [x.append(dt.datetime.strptime(t,change)) for t in df[colstr]]

    return x

def outlier(data):
    '''
    Objetivo: encontrar el numero de outliers de una columna de un dataset

    arg.
    ----
    data: columna del dataset

    ret.
    -------
    cantidad de outliers en esa columa (data)
    '''
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    ric=q3-q1
    
    outlier_1 = []

    # Fórmula de fuera de rango
    q_1_x = q1 - 1.5 * ric
    q_3_x = q3 + 1.5 * ric
    
    for i in data:
        if (i > q_3_x) | (i < q_1_x):
            outlier_1.append(i)
    return len(outlier_1)


                ###
# **Funciones para visualizaciones**
                ###

def grafica_tendencia(df_day,columns):
    '''
    Objetivo:

    args.
    ----
    ret.
    ----
    '''
    df_spn_365d = df_day[df_day.columns].rolling(window=365,center=True,min_periods=360).mean()
    df_spn_7d = df_day[df_day.columns].rolling(7, center=True).mean()
    if(len(columns)==1):
    
        fig, eje = plt.subplots()
        eje.plot(df_day['generation nuclear'], marker='.', markersize=2, color='0.6',
        linestyle='None', label='Diario')
        eje.plot(df_spn_7d['generation nuclear'], linewidth=2, label='Media deslizante semanal')
        eje.plot(df_spn_365d['generation nuclear'], color='0.2', linewidth=3,
        label='Tendencia (Media deslizante anual)')
        eje.legend()
        eje.set_xlabel('Año')
        eje.set_ylabel('Generacion (MWh)')
        eje.set_title('Tendencias')
    elif(len(columns)>1):
    
        fig,eje= plt.subplots()
        for i in columns:
            eje.plot(df_spn_365d[i],label=i)
            eje.set_ylim(0,15000)
            eje.legend()
            eje.set_ylabel('Producción (MWh)')
            eje.set_title('Tendencias en la generación')

    else:
        print('Fallo: pruebe a introducir columnas')
    
    return fig

def grafica_pie(df,columns,color=['burlywood','cadetblue','powderblue','slategray','seagreen', 'khaki','mediumaquamarine'],title='Energías'):
    '''
    Objetivo:

    args.
    ----
    ret.
    ----
    '''
    x1=df.sum()
    labels = columns
    sizes = (x1/x1.sum())*100
    explode = (0.1,)*df.shape[1] 


    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, colors=color,autopct='%1.1f%%',
            shadow=True, startangle=90,pctdistance=1.1)
    ax1.axis('equal')
    ax1.legend(loc='best',labels=labels)
    plt.title(title)

    plt.show()

    return fig1

def grafica_estacionalidad(df_day,columns,color='RdGy'):
    '''
    Objetivo:

    args.
    ----
    ret.
    ----
    '''
    fig, ejes = plt.subplots(len(columns), 1, sharex=True)
    for nombre, eje in zip(columns, ejes):
        sns.boxplot(data=df_day,x='month',y=nombre,ax=eje,palette=color)
        eje.set_title(nombre)
        if eje != ejes[-1]:
            eje.set_xlabel('')

    return fig

def grafica_acumulada(df,columns):
    '''
    Objetivo:

    args.
    ----
    ret.
    ----
    '''
    df.index=pd.to_datetime(df.index)
    df_spn_month_ = df[df.columns].resample('M').sum(min_count=28)
    df_spn_month_['total_cols']=df_spn_month_[df_spn_month_.columns[df_spn_month_.columns.str.startswith('generation')]].sum(axis=1)
    fig,eje = plt.subplots()
    eje.plot(df_spn_month_['total_cols'],color='black',label='generation total (MWh)')
    df_spn_month_[columns].plot.area(ax=eje,linewidth=0)
    eje.legend()
    eje.set_ylabel('Total Mensual (MWh)') 

    return fig


        ###
# **Funciones API:**
        ###


def api_ree(category,widget,headers=-1,start_date='2015-01-01T00:00',end_date='2015-01-31T00:00',time_trunc='day', geo_trunc=-1, geo_limit=-1, geo_ids=-1):
    '''
    Objetivo: obtener el .json de la api de reData.

    args.
    ----
    category: str; categoria a solicitar
    widget: str; subcategoria a solicitar
    headers: dict str; para añadir encabezados (opcional)

    start_date: str; fecha de inicio de los datos a recolectar; ejemplo: '2015-01-01T00:00' (opcional, aunque es necesaria para el request)
    end_date: str; fecha de fin de los datos a recolectar; ejemplo: '2015-01-01T00:00' (opcional, aunque es necesaria para el request)
    NOTA: hay limite de extracción. Se sugiere no extraer más de un año cada vez

    time_trunc: str; intervalo de tiempo a extraer, por defecto se ha elegido extracción diaria. (opcional, aunque es necesaria para el request)

    geo_trunc: str; para acotar zona geográfica. (opcional)
        geo_limit: str; por zona (opcional)
        geo_ids: str; por id (opcional)


    [https://www.ree.es/es/apidatos] - ir a esta página para más info.


    ret.
    ----
    json: devuelve el .json pedido
    '''
    url = "https://apidatos.ree.es/es/datos/"+category+'/'+widget

    if (geo_trunc!=-1):
        if (geo_limit!=-1):
            querystring = {"start_date":start_date,"end_date":end_date,"time_trunc":time_trunc,'geo_trunc':geo_trunc,'geo_limit':geo_limit}
        elif(geo_ids!=-1):
            querystring = {"start_date":start_date,"end_date":end_date,"time_trunc":time_trunc,'geo_trunc':geo_trunc,'geo_ids':geo_ids}
        else:
            print('Fallo: añadir geo_limit o geo_ids')
    else:
        querystring = {"start_date":start_date,"end_date":end_date,"time_trunc":time_trunc}


    if (headers!=-1):
        response = requests.request("GET", url,headers=headers, params=querystring)
    else:
        response = requests.request("GET", url, params=querystring)

    return response.json()

def extract_info(njson,values):

    '''
    Objetivo: extraer los datos de los valores.
    
    args.
    -----
    njson: json; archivo .json con datos anidados, los cuales se pretenden extraer y cambiar a un formato más amable.
    values: dict; indica la ubicación de los datos que se pretenden extraer.

    NOTA: para cada json será diferente y por lo tanto hay que visualizarlo a priori para indicar la posicion de los values.

    ret.
    ----
    res: list; lista de datos indicados en values para su extracción.
    '''
    res=[]
    
    for k,v in values.items():
        if type(v)==dict:
            for key,val in v.items():
                [res.append(njson[k][key][i]) for i in val]
        elif type(v)==list: 
            for l in v:
                if type(l)==dict:
                    for key,val in l.items():
                        [res.append(njson[k][key][i]) for i in val]
                else:
                    res.append(njson[k][l])

        else:
            res.append(njson[k][v])
    return res

def save_file(file,head,content):
    '''
    Objetivo: archivar datos en un .csv o .txt provenientes de una lista. 

    args.
    ----
    file: str; nombre del archivo.
    head: str; cabecera/columnas del archivo separadas por ';'.
    content: list; contenido a almacenar.

    ret.
    ----
    No devuelve nada, solo realiza el guardado. 
    '''
    # Comprobar que existe el fichero/ directorio
    #print(os.getcwd())
    ruta_dir=os.path.join(os.getcwd(),'data')
    os.makedirs(ruta_dir,exist_ok=True)
    ruta_file=os.path.join(ruta_dir,f'{file}')
    if os.path.exists(ruta_file):
        # Crear directorio y fichero si no existe
        with open(ruta_file, mode='a',newline='\n') as out:
            # Guardar la informacion
            [out.write(cont+';') if (content[-1]!=cont) else out.write(cont) for cont in content]
            out.write('\n')
        
    else:   
        with open(ruta_file, mode='w') as out:
            out.write(head+'\n')

def api_EDA_ev_demand(anios=[2015,2016,2017,2018],trunc='day',file='demand.txt'):
    '''
    Objetivo: extraer datos especificos, en este caso la evolución de la demanda, y archivarlos. 
    (No genérica a pesar de ser útil para extrapolar y trabajar con las funciones ya creadas)

    args.
    ----
    ret.
    ----
    '''
    lista_elem=[]
    for anio in anios:
        res=api_ree('demanda','evolucion',start_date=str(anio)+'-01-01T00:00',end_date=str(anio)+'-12-31T23:59',time_trunc=trunc)
        res=extract_info(res,{'included':{0:['attributes']}})
        
        for elem in range(0,len(res[0]['values'])):
            lista_elem.append(extract_info(res[0],{'values':{elem:['datetime','value']}}))
        lista_tun=[[str(lista_elem[i][0]),str(lista_elem[i][1])] for i in range(0,len(lista_elem))]

    for i in range(0,len(lista_tun)):
        save_file(file,'time;total_demand',lista_tun[i])

def api_EDA_p_inst(anios=[2015,2016,2017,2018],trunc='day',file='p_installed.txt'):
    '''
    Objetivo: extraer datos especificos, en este caso la potencia instalada en nuclear, eólica y solar, y archivarlos. 
    (No genérica a pesar de ser útil para extrapolar y trabajar con las funciones ya creadas)

    args.
    ----
    ret.
    ----
    '''
    lista_elem=[]

    for anio in anios:
        res=api_ree('generacion','potencia-instalada',start_date=str(anio)+'-01-01T00:00',end_date=str(anio)+'-12-31T23:59',time_trunc=trunc)
        res=extract_info(res,{'included':{2:['attributes'],10:['attributes'],11:['attributes']}})
        lista_tun=[[],]*len(res)
        for i in range(0,len(res)):
            for elem in range(0,len(res[i]['values'])):
                lista_elem.append(extract_info(res[i],{'values':{elem:['datetime','value']}}))
            lista_tun[i]=[[str(lista_elem[i][0]),str(lista_elem[i][1])] for i in range(0,len(lista_elem))]
            lista_elem=[]
        
        lista_tun2=[[lista_tun[0][i][0],lista_tun[0][i][1],lista_tun[1][i][1],lista_tun[2][i][1]] for i in range(0,len(lista_tun[0]))]
        for i in range(0,len(lista_tun2)):
            save_file(file,'time;nuclear;wind;solar',lista_tun2[i])


            ###
# **Funciones estadística:**
            ###

def test_normalidad(data, col,alpha=0.05):
    '''
    
    '''
    serie=data[col]
    # D'Agostino's K-squared test
    # ==============================================================================
    k2, p_value = ss.normaltest(serie)
    print("D'Agostino's K2 test:")
    print(f"\t Estadístico = {k2}, p-value = {p_value}")
    
    alpha=0.05
    if p_value < alpha:
        print(f"p-value < {alpha} -> Se rechaza H0 \n\t  No es posible asegurar que {col} sigue una distribución normal.")
    else:
        print(f"p-value > {alpha} -> Se acepta H0 \n\t   Es posible asegurar que {col} sigue una distribución normal.")
    print("*"*80)   

    # Histograma + curva normal teórica
    # ==============================================================================
    # Valores de la media (mu) y desviación típica (sigma) de los datos
    mu, sigma = ss.norm.fit(serie)

    # Valores teóricos de la normal en el rango observado
    x_hat = np.linspace(min(serie), max(serie), num=100)
    y_hat = ss.norm.pdf(x_hat, mu, sigma)

    # Gráfico
    fig, ax = plt.subplots()
    ax.plot(x_hat, y_hat, linewidth=2, color="slategray", label='normal teórica')
    ax.hist(x=serie, density=True, bins=80, alpha=0.5, color='slategray')
    ax.set_title(f'Distribución de {col}',fontsize = 10, fontweight = "bold")
    ax.set_xlabel(f'{col}')
    ax.set_ylabel('Densidad de probabilidad')
    ax.legend()

    # Gráfico Q-Q
    # ==============================================================================
    fig, ax = plt.subplots()
    sm.qqplot(serie, fit = True, line="s", alpha = 0.4, lw = 2, ax= ax, color='slategray')
    ax.set_title(f'Gráfico Q-Q de {col}', fontsize = 10, fontweight = "bold")
    ax.tick_params(labelsize = 7)

def test_homocedasticidad(col1,col2,alpha=0.05):
    '''
    
    '''
    # Levene test
    # ==============================================================================
    levene_test = ss.levene(col1, col2, center='median')
    
    # Fligner test
    # ==============================================================================
    fligner_test = ss.fligner(col1, col2, center='median')
    
    if (levene_test.pvalue>=alpha & fligner_test.pvalue>=alpha):
        print(f'p-value >= {alpha} -> Se acepta H0 \n\t   Es posible asegurar que los datos {col1,col2} proceden de distribuciones con la misma varianza (homocedasticidad).')
        res=[f'p-value >= {alpha} -> Se acepta H0 \n\t   Es posible asegurar que los datos {col1,col2} proceden de distribuciones con la misma varianza (homocedasticidad).',levene_test,fligner_test]
        print('Levene test: ',levene_test)
        print('Fligner test:',fligner_test)
        flag=1
    elif (levene_test.pvalue<alpha & fligner_test.pvalue<alpha):
        res=[f"p-value < {alpha} -> Se rechaza H0 \n\t  No es posible asegurar que los datos {col1,col2} proceden de distribuciones con la misma varianza (heterocedasticidad).",levene_test,fligner_test]
        flag=0
    else: 
        print(f"Los test difieren en su resultado, comprobar posibles causas")
        flag=-1
    return res,flag

def test_homocedasticidad_df(df):
    '''
    
    '''
    lista=list(df.select_dtypes(include='number').columns)
    n=0
    for i in range(len(lista)-1):
        for j in range(i+1,len(lista)):
            try:
                res,flag=test_homocedasticidad(df[lista[i]],df[lista[j]])
                if flag!=-1:
                    n+=flag
            except:
                pass
    if n==0:
        print('Todas las columnas son heterocedácticas entre si.')

def evaluar_hipotesis_df(df):
    lista=list(df.select_dtypes(include='number').columns)

    for i in range(len(lista)-1):
        for j in range(i+1,len(lista)):
            res=ss.f_oneway(np.array(df[lista[i]]),np.array(df[lista[j]])).pvalue
            

            if((res>=0.05) & (i!=j)):
                print("p-value >= {alpha} -> Se acepta H0 \n\t Se puede asegurar no introduce un efecto sobre la media total:",lista[i],lista[j])



def chi2_comp(df,var1,var2,alpha=0.05):
    '''
    
    '''
    stat, p, dof, expected=ss.chi2_contingency(pd.crosstab(df[var1],df[var2]))
    flag=0
    if p>=alpha:
        print(f'p-value >= {alpha} -> Se acepta H0 \n\t   Es posible asegurar que los datos {var1,var2} no tienen una relación significativa.')
        res=('','')
    else:
        print(f'p-value < {alpha} -> Se rechaza H0 \n\t   Es posible asegurar que los datos {var1,var2} tienen una relación de dependencia.')
        flag=1
        res=(var1,var2)

    return res,flag

# ---------
# CLASES 
# ---------