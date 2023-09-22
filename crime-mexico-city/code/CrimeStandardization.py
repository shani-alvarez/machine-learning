import numpy as np
import pandas as pd
from functools import reduce

# Class to standardize crime types according to the ICCS 2015
class Standardization:

    def __init__(self, data):
        self.data = data

    def homicidioTipo(self, df):
        df.loc[df.crime.str.contains('HOMICIDIO'), 'crimeCategory'] = 'HOMICIDIO'
        return df
    
    def homicidioModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CULPOSO)(?=.*HOMICIDIO)'), 'crimeType'] = 'HOMICIDIO CULPOSO'
        df.loc[df.crime.str.contains(r'(?!.*CULPOSO)(?=.*HOMICIDIO)'), 'crimeType'] = 'HOMICIDIO DOLOSO'
        return df
    
    def homicidioModo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ARMA DE FUEGO)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'ARMA DE FUEGO'
        df.loc[df.crime.str.contains(r'(?=.*ARMA BLANCA)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'ARMA BLANCA'
        df.loc[df.crime.str.contains(r'(?=.*GOLPES)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'GOLPES'
        df.loc[df.crime.str.contains(r'(?=.*AHORCAMIENTO)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'AHORCAMIENTO'
        df.loc[df.crime.str.contains(r'(?=.*TRÁNSITO VEHICULAR)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'TRANSITO VEHICULAR'
        df.loc[df.crime.str.contains(r'(?=.*ATROPELLADO)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'ATROPELLADO'
        df.loc[df.crime.str.contains(r'(?=.*COLISION)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'COLISION'
        df.loc[df.crime.str.contains(r'(?=.*CAIDA)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'CAIDA'
        df.loc[df.crime.str.contains(r'(?=.*PUNZO)(?=.*HOMICIDIO)'), 'crimeTypeViolence'] = 'INSTRUMENTO PUNZO CORTANTE'
        return df
    
    def roboTipo(self, df):
        df.loc[df.crime.str.contains(r'(?!.*SECUESTRO EXPRESS|VIOLACION|ILEGAL)(?=.*ROBO)'), 'crimeCategory'] = 'ROBO'
        return df

    def roboModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ROBO DE OBJETOS|ROBO DE DINERO|ROBO DE ACCESORIOS|ROBO DE DOCUMENTOS|ROBO DE ARMA|ROBO DE PLACA|ROBO DE FLUIDOS|ROBO DE ALHAJAS|ROBO DE ANIMALES|ROBO DE MAQUINARIA|ROBO DURANTE TRASLADO|ROBO DE CONTENEDORES|ROBO EN EVENTOS MASIVOS|ROBO DE MERCANCIA EN CONTENEDEROS|ROBO DE MERCANCIA A TRANSPORTISTA)'), 'crimeType'] = 'ROBO DE POSESIONES'
        df.loc[df.crime.str.contains(r'(?=.*ROBO DE VEHICULO|PESADO|ROBO DE MOTOCICLETA)'), 'crimeType'] = 'ROBO DE VEHICULO'
        df.loc[df.crime.str.contains(r'(?=.*ROBO A PASAJERO A BORDO DE TRANSPORTE PÚBLICO|ROBO A PASAJERO A BORDO DE METROBUS|ROBO A PASAJERO A BORDO DE PESERO|ROBO A PASAJERO A BORDO DE METRO|ROBO A PASAJERO EN ECOBUS|ROBO A PASAJERO EN RTP|ROBO A PASAJERO EN TREN|ROBO A PASAJERO EN TROLEBUS)'), 'crimeType'] = 'ROBO A PASAJERO A BORDO DE TRANSPORTE PUBLICO COLECTIVO'
        df.loc[df.crime.str.contains(r'(?=.*REPARTIDOR)(?=.*ROBO)'), 'crimeType'] = 'ROBO A REPARTIDOR'
        df.loc[df.crime.str.contains(r'(?=.*TRANSEUNTE)(?=.*ROBO)'), 'crimeType'] = 'ROBO A TRANSEUNTE EN VIA PUBLICA'
        df.loc[df.crime.str.contains(r'(?=.*ROBO A NEGOCIO|ROBO S/V DENTRO DE NEGOCIOS|ROBO A LOCALES|ROBO EN INTERIOR DE EMPRESA)'), 'crimeType'] = 'ROBO A NEGOCIO'
        df.loc[df.crime.str.contains(r'(?=.*CASA)(?=.*ROBO)'), 'crimeType'] = 'ROBO A CASA HABITACION'
        df.loc[df.crime.str.contains(r'(?=.*TAXI)(?=.*ROBO)'), 'crimeType'] = 'ROBO A BORDO DE TAXI'
        df.loc[df.crime.str.contains(r'(?=.*CONDUCTOR DE VEHICULO)(?=.*ROBO)'), 'crimeType'] = 'ROBO A PASAJERO/CONDUCTOR DE VEHICULO'
        df.loc[df.crime.str.contains(r'(?=.*TENTATIVA)(?=.*ROBO)'), 'crimeType'] = 'TENTATIVA DE ROBO'
        df.loc[df.crime.str.contains(r'(?=.*BANCARIA)(?=.*ROBO)'), 'crimeType'] = 'ROBO A SUCURSAL BANCARIA'
        df.loc[df.crime.str.contains(r'(?=.*OFICINA)(?=.*ROBO)'), 'crimeType'] = 'ROBO A OFICINA PUBLICA'
        df.loc[df.crime.str.contains(r'(?=.*AUTOBUS)(?=.*ROBO)'), 'crimeType'] = 'ROBO A PASAJERO EN AUTOBUS FORANEO'
        df.loc[df.crime.str.contains(r'(?=.*AUTOBÚS)(?=.*ROBO)'), 'crimeType'] = 'ROBO A PASAJERO EN AUTOBUS FORANEO'
        return df

    def roboModo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CON VIOLENCIA)(?=.*ROBO)'), 'crimeTypeViolence'] = 'CON VIOLENCIA'
        df.loc[df.crime.str.contains(r'(?=.*C/V)(?=.*ROBO)'), 'crimeTypeViolence'] = 'CON VIOLENCIA'
        df.loc[df.crime.str.contains(r'(?=.*SIN VIOLENCIA)(?=.*ROBO)'), 'crimeTypeViolence'] = 'SIN VIOLENCIA'
        df.loc[df.crime.str.contains(r'(?=.*S/V)(?=.*ROBO)'), 'crimeTypeViolence'] = 'SIN VIOLENCIA'
        return df


    def lesionesTipo(self, df):
        df.loc[df.crime.str.contains('LESIONES'), 'crimeCategory'] = 'LESIONES'
        return df
        
    def lesionesModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CULPOSAS)(?=.*LESIONES)'), 'crimeType'] = 'LESIONES CULPOSAS'
        df.loc[df.crime.str.contains(r'(?=.*INTENCIONALES)(?=.*LESIONES)'), 'crimeType'] = 'LESIONES DOLOSAS'
        df.loc[df.crime.str.contains(r'(?=.*DOLOSAS)(?=.*LESIONES)'), 'crimeType'] = 'LESIONES DOLOSAS'
        return df
        
    def lesionesModo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ARMA DE FUEGO)(?=.*LESIONES)'), 'crimeTypeViolence'] = 'ARMA DE FUEGO'
        df.loc[df.crime.str.contains(r'(?=.*ARMA BLANCA)(?=.*LESIONES)'), 'crimeTypeViolence'] = 'ARMA BLANCA'
        df.loc[df.crime.str.contains(r'(?=.*GOLPES)(?=.*LESIONES)'), 'crimeTypeViolence'] = 'GOLPES'
        df.loc[df.crime.str.contains(r'(?=.*ACCIDENTE LABORAL)(?=.*LESIONES)'), 'crimeTypeViolence'] = 'ACCIDENTE LABORAL'
        df.loc[df.crime.str.contains(r'(?=.*TRANSITO)(?=.*LESIONES)'), 'crimeTypeViolence'] = 'TRANSITO VEHICULAR'
        df.loc[df.crime.str.contains(r'(?=.*VEHICULO)(?=.*LESIONES)'), 'crimeTypeViolence'] = 'TRANSITO VEHICULAR'
        df.loc[df.crime.str.contains(r'(?=.*QUEMADURAS)(?=.*LESIONES)'), 'crimeTypeViolence'] = 'QUEMADURAS'
        df.loc[df.crime.str.contains(r'(?=.*CAIDA)(?=.*LESIONES)'), 'crimeTypeViolence'] = 'CAIDA'
        return df
        
    def violfamTipo(self, df):
        df.loc[df.crime.str.contains('VIOLENCIA FAMILIAR'), 'crimeCategory'] = 'VIOLENCIA FAMILIAR'
        return df
    
    def violfamModalidad(self, df):
        df.loc[df.crime.str.contains('VIOLENCIA FAMILIAR'), 'crimeType'] = 'VIOLENCIA FAMILIAR'
        return df
    
    def violacionTipo(self, df):
        df.loc[df.crime.str.contains(r'(?!.*CORRESPONDENCIA)(?=.*VIOLACION)'), 'crimeCategory'] = 'VIOLACION'   
        return df
        
    def violacionModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?!.*CORRESPONDENCIA)(?=.*VIOLACION)'), 'crimeType'] = 'VIOLACION SIMPLE'
        df.loc[df.crime.str.contains(r'(?=.*EQUIPARADA)(?=.*VIOLACION)'), 'crimeType'] = 'VIOLACION EQUIPARADA'
        df.loc[df.crime.str.contains(r'(?=.*TENTATIVA)(?=.*VIOLACION)'), 'crimeType'] = 'TENTATIVA DE VIOLACION'
        return df
    
    def privTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*SECUESTRO|PRIVACION|PRIV\. ILEGAL|SUSTRACCION|SUSTRACCIÓN|SUSTRACCIÃ“N|ROBO DE INFANTE|ENTREGA ILEGITIMA|DESAPARICION FORZADA|RETENCIÓN DE MENORES|TRAFICO DE INFANTES)'), 'crimeCategory'] = 'PRIVACION DE LA LIBERTAD PERSONAL' 
        return df
        
    def privModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*EXPRESS)(?=.*SECUESTRO)'), 'crimeType'] = 'SECUESTRO'
        df.loc[df.crime.str.contains(r'(?=.*PLAGIO)(?=.*SECUESTRO)'), 'crimeType'] = 'SECUESTRO'
        df.loc[df.crime.str.contains(r'(?=.*SECUESTRO)'), 'crimeType'] = 'SECUESTRO'
        df.loc[df.crime.str.contains(r'(?=.*PRIVACION|PRIV\. ILEGAL)'), 'crimeType'] = 'PRIVACION ILEGAL DE LA LIBERTAD PERSONAL'
        df.loc[df.crime.str.contains(r'(?=.*SUSTRACCION)(?=.*MENORES)'), 'crimeType'] = 'SUSTRACCION DE MENORES'   
        df.loc[df.crime.str.contains(r'(?=.*SUSTRACCIÓN)(?=.*MENORES)'), 'crimeType'] = 'SUSTRACCION DE MENORES'
        df.loc[df.crime.str.contains(r'(?=.*SUSTRACCIÃ“N)(?=.*MENORES)'), 'crimeType'] = 'SUSTRACCION DE MENORES'
        df.loc[df.crime.str.contains(r'(?=.*RETENCIÓN)(?=.*MENORES)'), 'crimeType'] = 'RETENCIÓN DE MENORES'
        df.loc[df.crime.str.contains(r'(?=.*ROBO)(?=.*INFANTE)'), 'crimeType'] = 'SUSTRACCION DE MENORES'
        df.loc[df.crime.str.contains(r'(?=.*ENTREGA ILEGITIMA)'), 'crimeType'] = 'ENTREGA ILEGITIMA DE UN MENOR'   
        df.loc[df.crime.str.contains(r'(?=.*DESAPARICION FORZADA)'), 'crimeType'] = 'DESAPARICION FORZADA'
        df.loc[df.crime.str.contains(r'(?=.*TRAFICO DE INFANTES)'), 'crimeType'] = 'TRAFICO DE INFANTES' 
        return df
    
    def feminicidioTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*FEMINICIDIO)'), 'crimeCategory'] = 'FEMINICIDIO'  
        return df

    def feminicidioModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*FEMINICIDIO)'), 'crimeType'] = 'FEMINICIDIO'
        return df

    def feminicidioModo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ARMA BLANCA)(?=.*FEMINICIDIO)'), 'crimeTypeViolence'] = 'ARMA BLANCA'
        df.loc[df.crime.str.contains(r'(?=.*ARMA DE FUEGO)(?=.*FEMINICIDIO)'), 'crimeTypeViolence'] = 'ARMA DE FUEGO'
        df.loc[df.crime.str.contains(r'(?=.*GOLPES)(?=.*FEMINICIDIO)'), 'crimeTypeViolence'] = 'GOLPES' 
        return df
        
    def abusoconfTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ABUSO DE CONFIANZA)'), 'crimeCategory'] = 'ABUSO DE CONFIANZA'
        return df
    
    def abusoconfModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ABUSO DE CONFIANZA)'), 'crimeType'] = 'ABUSO DE CONFIANZA'
        return df
    
    def abusosexTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ABUSO SEXUAL|ESTUPRO)'), 'crimeCategory'] = 'ABUSO SEXUAL'
        return df
    
    def abusosexModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ABUSO SEXUAL|ESTUPRO)'), 'crimeType'] = 'ABUSO SEXUAL'
        return df
    
    def acososexTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ACOSO SEXUAL)'), 'crimeCategory'] = 'ACOSO SEXUAL'
        return df
    
    def acososexModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ACOSO SEXUAL)'), 'crimeType'] = 'ACOSO SEXUAL'  
        return df
        
    def fraudeTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*FRAUDE)'), 'crimeCategory'] = 'FRAUDE'
        return df
    
    def fraudeModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*FRAUDE)'), 'crimeType'] = 'FRAUDE'
        return df

    def corrmenTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CORRUPCION|CORRUPCIÓN)(?=.*MENORES)'), 'crimeCategory'] = 'CORRUPCION DE MENORES'
        return df
    
    def corrmenModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CORRUPCION|CORRUPCIÓN)(?=.*MENORES)'), 'crimeType'] = 'CORRUPCION DE MENORES'
        return df
    
    def torturaTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*TORTURA)'), 'crimeCategory'] = 'TORTURA'
        return df
    
    def torturaModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*TORTURA)'), 'crimeType'] = 'TORTURA'
        return df
    
    def trataTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*TRATA DE PERSONAS)'), 'crimeCategory'] = 'TRATA DE PERSONAS'
        return df
    
    def trataModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*TRATA DE PERSONAS)'), 'crimeType'] = 'TRATA DE PERSONAS'
        return df

    def explolabTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*EXPLOTACION|EXPLOTACIÓN)'), 'crimeCategory'] = 'EXPLOTACION LABORAL'
        return df
    
    def explolabModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*EXPLOTACION|EXPLOTACIÓN)'), 'crimeType'] = 'EXPLOTACION DE MENOR O DISCAPACITADO'
        return df
    
    def explosexTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*LENOCINIO|PORNOGRAFIA INFANTIL)'), 'crimeCategory'] = 'EXPLOTACION SEXUAL'
        return df
    
    def explosexModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*LENOCINIO)'), 'crimeType'] = 'LENOCINIO'
        df.loc[df.crime.str.contains(r'(?=.*PORNOGRAFIA INFANTIL)'), 'crimeType'] = 'PORNOGRAFIA INFANTIL'  
        return df
        
    def narcomenudeoTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*NARCOMENUDEO)'), 'crimeCategory'] = 'NARCOMENUDEO'
        return df

    def narcomenudeoModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*SIMPLE)(?=.*NARCOMENUDEO)'), 'crimeType'] = 'NARCOMENUDEO POSESION SIMPLE'
        df.loc[df.crime.str.contains(r'(?!.*SIMPLE)(?=.*NARCOMENUDEO)'), 'crimeType'] = 'NARCOMENUDEO POSESION CON FINES DE VENTA, COMERCIO Y SUMINISTRO'
        return df

    def extorsionTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*EXTORSION)'), 'crimeCategory'] = 'EXTORSION'
        return df

    def extorsionModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?!.*TENTATIVA)(?=.*EXTORSION)'), 'crimeType'] = 'EXTORSION'
        df.loc[df.crime.str.contains(r'(?=.*TENTATIVA)(?=.*EXTORSION)'), 'crimeType'] = 'TENTATIVA DE EXTORSION' 
        return df
        
    def amenazasTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*AMENAZAS|INTIMIDACION|COBRANZA|EXHORTOS|ALLANAMIENTO)'), 'crimeCategory'] = 'DELITOS CONTRA LA PAZ Y SEGURIDAD DE LAS PERSONAS' 
        return df
        
    def amenazasModalidad(self, df):
        df.loc[df.crime.str.contains(r'^(?=.*AMENAZAS)'), 'crimeType'] = 'AMENAZAS'
        df.loc[df.crime.str.contains(r'^(?=.*INTIMIDACION)'), 'crimeType'] = 'INTIMIDACION'
        df.loc[df.crime.str.contains(r'^(?=.*COBRANZA)'), 'crimeType'] = 'COBRANZA ILEGITIMA'
        df.loc[df.crime.str.contains(r'^(?=.*EXHORTOS)'), 'crimeType'] = 'EXHORTOS'
        df.loc[df.crime.str.contains(r'^(?=.*ALLANAMIENTO)'), 'crimeType'] = 'ALLANAMIENTO' 
        return df
        
    def ordensexTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*PORNOGRAFIA|PORNOGRAFÍA|INCESTO|BIGAMIA|CONTRA LA INTIMIDAD SEXUAL)'), 'crimeCategory'] = 'DELITOS QUE ATENTAN CONTRA LOS ESTANDARES SEXUALES DEL ORDEN PUBLICO'
        return df
    
    def ordensexModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*PORNOGRAFIA|PORNOGRAFÍA|INCESTO|BIGAMIA|CONTRA LA INTIMIDAD SEXUAL)'), 'crimeType'] = 'DELITOS QUE ATENTAN CONTRA LOS ESTANDARES SEXUALES DEL ORDEN PUBLICO'
        return df
        
    def autoridadTipo(self, df):
        df.loc[df.crime.str.contains(r'^(?=.*QUEBRANTAMIENTO|CONTRA FUNCIONARIOS|RESISTENCIA DE PARTICULARES|DESOBEDIENCIA|DESOBEDENCIA|OPOSICION|ULTRAJES|DESACATO)'), 'crimeCategory'] = 'DELITOS CONTRA LA AUTORIDAD'
        return df
    
    def autoridadModalidad(self, df):
        df.loc[df.crime.str.contains(r'^(?=.*QUEBRANTAMIENTO|CONTRA FUNCIONARIOS|RESISTENCIA DE PARTICULARES|DESOBEDIENCIA|DESOBEDENCIA|OPOSICION|ULTRAJES|DESACATO)'), 'crimeType'] = 'DELITOS CONTRA LA AUTORIDAD'
        return df
    
    def falsedadTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*FALSIFICACION|FALSIFICACIÓN|FALSIFICACIÃ“N|USURPACION|USURPACIÓN|USURPACIÃ“N|USO INDEBIDO|FALSEDAD|FALSO|VARIACION)'), 'crimeCategory'] = 'FALSEDAD Y USURPACION DE FUNCIONES PUBLICAS'
        return df
        
    def falsedadModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*FALSIFICACION|FALSIFICACIÓN|USURPACION|USURPACIÓN|USO INDEBIDO|FALSEDAD|FALSO|VARIACION)'), 'crimeType'] = 'FALSEDAD Y USURPACION DE FUNCIONES PUBLICAS'
        return df
    
    def leyArmasTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*PORTACIÓN|PORTACION|EXPLOSIVOS|OBJETOS APTOS PARA AGREDIR)'), 'crimeCategory'] = 'LEY FEDERAL DE ARMAS DE FUEGO Y EXPLOSIVOS'
        return df
               
    def leyArmasModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*PORTACION|PORTACIÓN)(?=.*PROHIB)'), 'crimeType'] = 'PORTACION ARMA/PROHIB'
        df.loc[df.crime.str.contains(r'(?=.*PORTACION|PORTACIÓN)(?=.*FUEGO)'), 'crimeType'] = 'PORTACION DE ARMA DE FUEGO'
        df.loc[df.crime.str.contains(r'(?=.*EXPLOSIVOS)'), 'crimeType'] = 'LEY GENERAL DE EXPLOSIVOS'
        df.loc[df.crime.str.contains(r'(?=.*OBJETOS APTOS PARA AGREDIR)'), 'crimeType'] = 'PORTACION, FABRICACION E IMPORTACION DE OBJETOS APTOS PARA AGREDIR'
        return df
               
    def seguridadTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*DISPAROS|PANDILLA|EVASION DE PRESOS|MOTIN|ASOCIACION DELICTUOSA)'), 'crimeCategory'] = 'DELITOS CONTRA LA SEGURIDAD PUBLICA'
        return df
    
    def seguridadModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*DISPAROS|PANDILLA|EVASION DE PRESOS|MOTIN|ASOCIACION DELICTUOSA)'), 'crimeType'] = 'DELITOS CONTRA LA SEGURIDAD PUBLICA'
        return df
    
    def corrupcionTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ABUSO DE AUTORIDAD|COHECHO|EJERCICIO ILEGAL|EJERCICIO INDEBIDO|NEGACION DEL SERVICIO|COALICION|COALICIÓN|INFLUENCIA|ENRIQUECIMIENTO|PECULADO|CONCUSION|EJERCICIO ABUSIVO DE FUNCIONES|COACCION DE SERVIDORES PUBLICOS)'), 'crimeCategory'] = 'DELITOS POR HECHOS DE CORRUPCION'
        return df

    def corrupcionModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*ABUSO DE AUTORIDAD|COHECHO|EJERCICIO ILEGAL|EJERCICIO INDEBIDO|NEGACION DEL SERVICIO|COALICION|COALICIÓN|INFLUENCIA|ENRIQUECIMIENTO|PECULADO|CONCUSION|EJERCICIO ABUSIVO DE FUNCIONES|COACCION DE SERVIDORES PUBLICOS)'), 'crimeType'] = 'DELITOS POR HECHOS DE CORRUPCION'
        return df

    def honorTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CALUMNIAS|DIFAMACION)'), 'crimeCategory'] = 'DELITOS CONTRA EL HONOR'
        return df
               
    def honorModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CALUMNIAS|DIFAMACION)'), 'crimeType'] = 'DELITOS CONTRA EL HONOR'
        return df
               
    def comunicacionTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CORRESPONDENCIA|COMUNICACION|COMUNICACIÓN)'), 'crimeCategory'] = 'DELITOS EN MATERIA DE VIAS DE COMUNICACION Y CORRESPONDENCIA'
        return df
    
    def comunicacionModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CORRESPONDENCIA|COMUNICACION|COMUNICACIÓN)'), 'crimeType'] = 'DELITOS EN MATERIA DE VIAS DE COMUNICACION Y CORRESPONDENCIA'
        return df
    
    def saludTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CONTAGIO|CONTRA LA SALUD|PROCREACION|INSEMINACION|ESTERILIZACION)'), 'crimeCategory'] = 'DELITOS CONTRA LA SALUD'
        return df
    
    def saludModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*CONTAGIO|CONTRA LA SALUD)'), 'crimeType'] = 'DELITOS CONTRA LA SALUD'
        df.loc[df.crime.str.contains(r'(?=.*PROCREACION|INSEMINACION|ESTERILIZACION)'), 'crimeType'] = 'DELITOS CONTRA LOS DERECHOS REPRODUCTIVOS'
        return df
    
    def ambienteTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*AMBIENTALES|AMBIENTAL|CONTAMINACIÓN|TALA)'), 'crimeCategory'] = 'DELITOS CONTRA EL AMBIENTE'
        return df
    
    def ambienteModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*AMBIENTALES|AMBIENTAL|CONTAMINACIÓN|TALA)'), 'crimeType'] = 'DELITOS CONTRA EL AMBIENTE'
        return df
    
    def propiedadTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*DAÑO|DESPOJO|DAÃ‘O)'), 'crimeCategory'] = 'DAÑO EN PROPIEDAD AJENA'
        return df
    
    def propiedadModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*DAÑO|DESPOJO|DAÃ‘O)'), 'crimeType'] = 'DAÑO EN PROPIEDAD AJENA'
        return df
    
    def otrosTipo(self, df):
        df.loc[df.crime.str.contains(r'(?=.*PERDIDA DE LA VIDA|DENUNCIA|ADMINISTRACION DE JUSTICIA|ABANDONO DE PERSONA|EXPOSICION DE MENORES|ABORTO|INSOLVENCIA|POSESION DE VEHICULO ROBADO|OTROS|INHUMACIONES|INHUMACION|PERSONAS EXTRAVIADAS|DDH|OMISION|CONTRA EL CUMPLIMIENTO|SUICIDIO|USO DE SUELO|URBANA|PAZ|APOLOGIA|MALTRATO ANIMAL|ENCUBRIMIENTO|DELITOS DE ABOGADOS|RESPONSABILIDAD|SECRETOS|PROCEDENCIA ILICITA|PROCEDENCIA ILEGAL|SABOTAJE|DISCRIMINACION|ELECTORALES|ESTADO CIVIL|UTILIZACION INDEBIDA DE LA VIA PUBLICA)'), 'crimeCategory'] = 'OTROS DELITOS DEL FUERO COMUN'
        return df
    
    def otrosModalidad(self, df):
        df.loc[df.crime.str.contains(r'(?=.*PERDIDA DE LA VIDA|DENUNCIA|ADMINISTRACION DE JUSTICIA|ABANDONO DE PERSONA|EXPOSICION DE MENORES|ABORTO|INSOLVENCIA|POSESION DE VEHICULO ROBADO|OTROS|INHUMACIONES|INHUMACION|PERSONAS EXTRAVIADAS|DDH|OMISION|CONTRA EL CUMPLIMIENTO|SUICIDIO|USO DE SUELO|URBANA|PAZ|APOLOGIA|MALTRATO ANIMAL|ENCUBRIMIENTO|DELITOS DE ABOGADOS|RESPONSABILIDAD|SECRETOS|PROCEDENCIA ILICITA|PROCEDENCIA ILEGAL|SABOTAJE|DISCRIMINACION|ELECTORALES|ESTADO CIVIL|UTILIZACION INDEBIDA DE LA VIA PUBLICA)'), 'crimeType'] = 'OTROS DELITOS DEL FUERO COMUN'
        return df
        
    def errors(self, df):
        errors_category = df[df.crimeCategory.isna()]
        print("****************************** Errors in data categorization ********************************", '\n')
        print("Errors in the categories: " + str(errors_category.crime.unique()) + "\n")
        errors_type = df[df.crimeType.isna()]
        print("Errors in the types: " + str(errors_type.crime.unique()) + "\n")
        
    def function_composition(self, *func):
        def composition(f, g): 
            return lambda x : f(g(x)) 
        return reduce(composition, func, lambda x : x) 
    
    
    def categorizeCrimeData(self):
        standardization = self.function_composition(
        self.homicidioTipo, self.homicidioModalidad, self.homicidioModo, self.roboTipo, self.roboModalidad, self.roboModo, self.lesionesTipo, self.lesionesModalidad, self.lesionesModo, self.violfamTipo, self.violfamModalidad, self.violacionTipo, self.violacionModalidad, self.privTipo, self.privModalidad, self.feminicidioTipo, self.feminicidioModalidad, self.feminicidioModo, self.abusoconfTipo, self.abusoconfModalidad, self.abusosexTipo, self.abusosexModalidad, self.acososexTipo, self.acososexModalidad, self.fraudeTipo, self.fraudeModalidad, self.corrmenTipo, self.corrmenModalidad, self.torturaTipo, self.torturaModalidad, self.trataTipo, self.trataModalidad, self.explolabTipo, self.explolabModalidad, self.explosexTipo, self.explosexModalidad, self.narcomenudeoTipo, self.narcomenudeoModalidad, self.extorsionTipo, self.extorsionModalidad, self.amenazasTipo, self.amenazasModalidad, self.ordensexTipo, self.ordensexModalidad, self.autoridadTipo, self.autoridadModalidad, self.falsedadTipo, self.falsedadModalidad, self.seguridadTipo, self.seguridadModalidad, self.corrupcionTipo, self.corrupcionModalidad, self.comunicacionTipo, self.comunicacionModalidad, self.saludTipo, self.saludModalidad, self.ambienteTipo, self.ambienteModalidad, self.propiedadTipo, self.propiedadModalidad, self.otrosTipo, self.otrosModalidad, self.leyArmasTipo, self.leyArmasModalidad, self.honorTipo, self.honorModalidad)
        df = standardization(self.data)
        self.errors(df)        
        return df
        

