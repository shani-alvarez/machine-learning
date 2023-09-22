# Data pre-processing class for the crime and victims datasets

import pandas as pd
import geopandas as gp
import numpy as np
import os
import sys
import glob
from CrimeStandardization import Standardization

class PreProcessingCrimeData:
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_directory = os.getcwd()
        self.raw_data_directory = os.path.normpath(os.path.join(self.current_directory, '../data/raw'))
        self.proc_data_directory = os.path.normpath(os.path.join(self.current_directory, '../data/preprocessed'))
        if dataset == 'crime':
            self.data = pd.read_csv(self.raw_data_directory + '/crime_reports_feb2022.csv')
            print("Dataset read: " + str(self.raw_data_directory) + '/crime_reports_feb2022.csv' + '\n')
        elif dataset == 'victims':
            self.data = pd.read_csv(self.raw_data_directory + '/victims_reports_feb2022.csv')
            print("Dataset read: " + str(self.raw_data_directory) + '/victims_reports_feb2022.csv' + '\n')
        else:
            print('Choose which dataset you want to pre-process.')
        
    def getDirectories(self):
        print("Current directory: ", self.current_directory)
        print("Raw data directory: ", self.raw_data_directory)
        print("Pre-processed data directory: ", self.proc_data_directory)
        print()
        
    def datasetInformation(self):
        print('Number of observations: ' + str(self.data.shape[0]) + '\n')
        print('Number of attributes: ' + str(self.data.shape[1]) + '\n')
        print("**************************** Raw dataset information *****************************", '\n')
        print(self.data.info(), '\n')
        
    def translateCrimeColumnNames(self, data):
        cols_translation = {'ao_hechos': 'year_events', 'mes_hechos': 'month_events', 'fecha_hechos': 'date_events', 'ao_inicio': 'year_start', 
                            'mes_inicio': 'month_start', 'fecha_inicio': 'date_start', 'delito': 'crime', 'fiscalia': 'attorneys_office', 
                            'agencia':'agency', 'unidad_investigacion': 'investigation_unit', 'categoria_delito': 'crime_category', 
                            'calle_hechos': 'street_events', 'calle_hechos2': 'street_events2', 'colonia_hechos': 'neighborhood_events', 
                            'alcaldia_hechos': 'borough_events', 'competencia': 'competence', 'longitud': 'longitude', 'latitud': 'latitude',
                           'tempo': 'tempo'}
        df = data.copy()
        df.columns = list(cols_translation.values())
        return df
    
    def translateVictimsColumnNames(self, data):
        cols_translation = {'idCarpeta': 'id_report', 'Año_inicio': 'year_start', 'Mes_inicio': 'month_start', 'FechaInicio': 'date_start', 'Delito': 'crime', 'Categoria': 'crime_category', 'Sexo': 'sex', 'Edad': 'age', 'TipoPersona': 'type_person', 'CalidadJuridica': 'legal_quality', 'competencia': 'competence', 'Año_hecho': 'year_events', 'Mes_hecho': 'month_events', 'FechaHecho': 'date_events', 'HoraHecho': 'time_events', 'HoraInicio': 'time_start', 'AlcaldiaHechos': 'borough_events', 'ColoniaHechos': 'neighborhood_events', 'Calle_hechos': 'street_events', 'Calle_hechos2': 'street_events2', 'latitud': 'latitude', 'longitud': 'longitude'}
        df = data.copy()
        df.columns = list(cols_translation.values())
        return df
    
    def filterByYear(self, data, ini_year, end_year):
        years = list(range(ini_year, end_year + 1))
        df = data[data.year_events.isin(years)].reset_index(drop=True)
        return df
    
    def narrowDataValues(self, data):
        #Check values of boroughs
        boroughs = ['ALVARO OBREGON', 'AZCAPOTZALCO', 'BENITO JUAREZ', 'COYOACAN', 'CUAJIMALPA DE MORELOS', 'CUAUHTEMOC', 
                    'GUSTAVO A MADERO', 'IZTACALCO', 'IZTAPALAPA', 'LA MAGDALENA CONTRERAS', 'MIGUEL HIDALGO', 'MILPA ALTA', 
                    'TLAHUAC', 'TLALPAN', 'VENUSTIANO CARRANZA', 'XOCHIMILCO']
        df = data[data.borough_events.isin(boroughs)].reset_index(drop=True)
        print("List of Mexico City's boroughs: ", list(df.borough_events.unique()))
        print()
        return df
            
    def standardizeCrimes(self, data):
        cs = Standardization(data)
        df = cs.categorizeCrimeData()
        return df
    
    def lowercaseTextCrimeData(self, data):
        categorical_columns = ['month_events', 'month_start', 'crime', 'attorneys_office', 'agency', 'investigation_unit', 
                               'crime_category', 'street_events', 'street_events2', 'neighborhood_events', 'borough_events', 
                               'competence', 'crimeType', 'crimeCategory', 'crimeTypeViolence']
        df = data.copy()
        df[categorical_columns] = df[categorical_columns].astype("string")
        df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.lower())
        return df
    
    def lowercaseTextVictimsData(self, data):
        categorical_columns = ['month_start', 'crime', 'crime_category', 'sex', 'type_person', 'legal_quality', 'competence', 'month_events', 'borough_events', 'neighborhood_events', 'street_events', 'street_events2', 'crimeType', 'crimeCategory', 'crimeTypeViolence']
        df = data.copy()
        df[categorical_columns] = df[categorical_columns].astype("string")
        df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.lower())
        return df
    
    # Select crime type
    def selectCrimeType(self, data, crime_list):
        df = data[data.crimeType.isin(crime_list)]
        return df
    
    def selectFeatures(self, data, features_list):
        df = data[features_list].reset_index(drop=True)
        return df
    
    def handleMissingData(self, data):
        df = data.dropna(subset=['longitude'])
        print("Missing values: ", '\n')
        print(df.isna().sum(), '\n')
        return df
    
    def createExtraFeatures(self, data):
        df = data.copy()
        df["date_events"] = pd.to_datetime(df["date_events"])
        df["year_event"] = df["date_events"].dt.year
        df["month_event"] = df["date_events"].dt.month
        df["day_event"] = df["date_events"].dt.day
        df["hour_event"] = df["date_events"].dt.hour
        return df
    
    def fillCrimeViolence(self, data):
        df = data.copy()
        df["crimeTypeViolence"].fillna('sin registro', inplace = True)
        return df
    
    def fillAgeSex(self, data):
        df = data.copy()
        #df["age"].fillna('sin registro', inplace = True)
        df["sex"].fillna('sin registro', inplace = True)
        return df
    
    def checkValuesTypes(self, data):
        df = data.copy()
        df["crimeType"] = df["crimeType"].astype('category')
        df["crimeTypeViolence"] = df["crimeTypeViolence"].astype('category')
        df["neighborhood_events"] = df["neighborhood_events"].astype('category')
        return df
    
    def checkSexAgeValues(self, data):
        df = data.copy()
        #df["age"] = df["age"].astype('int32')
        df["sex"] = df["sex"].astype('category')
        return df
   
    def preProcessCrime(self, ini_year, end_year, crime_list, features_list):
        self.getDirectories()
        self.datasetInformation()
        df = self.translateCrimeColumnNames(self.data) 
        df = self.filterByYear(df, ini_year, end_year)
        df = self.narrowDataValues(df)
        df2 = self.standardizeCrimes(df)
        df3 = self.lowercaseTextCrimeData(df2) 
        df3 = self.selectCrimeType(df3, crime_list)
        df3 = self.selectFeatures(df3, features_list) 
        df3 = self.handleMissingData(df3) 
        df3 = self.createExtraFeatures(df3) 
        df3 = self.fillCrimeViolence(df3) 
        df_final = self.checkValuesTypes(df3)  
        df_final.to_csv(self.proc_data_directory + '/preprocessed_crime_data.csv', index=False)
        print("****************************** Pre-processed dataset information ********************************")
        print(df_final.info(), '\n')
        return df_final
    
    def preProcessVictims(self, ini_year, end_year, crime_list, features_list):
        self.getDirectories()
        self.datasetInformation()
        df = self.translateVictimsColumnNames(self.data) 
        df = self.filterByYear(df, ini_year, end_year)
        df = self.narrowDataValues(df)
        df2 = self.standardizeCrimes(df)
        df3 = self.lowercaseTextVictimsData(df2) 
        df3 = self.selectCrimeType(df3, crime_list)
        df3 = self.selectFeatures(df3, features_list) 
        df3 = self.handleMissingData(df3) 
        df3 = self.createExtraFeatures(df3) 
        df3 = self.fillCrimeViolence(df3) 
        df3 = self.fillAgeSex(df3)
        df3 = self.checkSexAgeValues(df3)
        df_final = self.checkValuesTypes(df3)  
        df_final.to_csv(self.proc_data_directory + '/preprocessed_victims_data.csv', index=False)
        print("****************************** Pre-processed dataset information ********************************", '\n')
        print(df_final.info(), '\n')
        return df_final
    
    def loadProcessedDatasets(self):
        list_shapefiles = []
        if self.dataset == 'crime':
            df = pd.read_csv(self.proc_data_directory + '/preprocessed_crime_data.csv')
            print("Dataset read: " + str(self.proc_data_directory) + '/preprocessed_crime_data.csv' + '\n')
        elif self.dataset == 'victims':
            df = pd.read_csv(self.proc_data_directory + '/preprocessed_victims_data.csv')
            print("Dataset read: " + str(self.proc_data_directory) + '/preprocessed_victims_data.csv' + '\n')
        else:
            print('Choose which dataset you want to load.')
        return df
    
    
    
class PreProcessingShapefiles:
    
    def __init__(self):
        self.current_directory = os.getcwd()
        self.data_directory = os.path.normpath(os.path.join(self.current_directory, '../data/raw/shapefiles/'))
        self.proc_data_directory = os.path.normpath(os.path.join(self.current_directory, '../data/preprocessed'))
    
    def loadFiles(self):
        #Load datasets
        path = self.data_directory
        list_files = []
        list_dir = []
        count = 0

        for path in glob.glob(f'{self.data_directory}/*/'):
            list_dir.append(path)

        for folder in list_dir:
            shape_files = glob.glob(os.path.join(folder, "*.shp"))
            print("Loading dataset.............................", "\n")
            for file in shape_files:
                df = gp.read_file(file)
                list_files.append(df)
                print(count)
                print('Folder:', folder)
                print('File Name:', file.split("\\")[-1])
                print('Rows: ' + str(df.shape[0]) + ', columns: ' + str(df.shape[1]), '\n')
                count += 1
        return list_files
    
    def reprojectShapefiles(self, list_shapefiles):
        reprojected_shapefiles = []
        for gpdf in list_shapefiles:
            gpdf = gpdf.to_crs("EPSG:32614")
            reprojected_shapefiles.append(gpdf)
            if gpdf.crs != 'EPSG:32614':
                print('Wrong reprojection.')
            else:
                pass
        print("All shapefiles were successfully reprojected.")
        return reprojected_shapefiles
    
    def countPointsInPolygon(self, list_shapefiles, polygon_shapefile):
        cols_names = ['cablebus_sta', 'commer_venues', 'health_centres', 'metro_sta', 'pmarkets', 'pparking','hospitals', 
                      'train_sta', 'trolebus_sta']
        gdf = polygon_shapefile.copy()
        for i in range(0, len(list_shapefiles)):
            gdf = gdf.merge(gdf.sjoin(list_shapefiles[i], predicate='contains').groupby('key_neighbor').size().rename(cols_names[i]).reset_index(), how='left').fillna(0)
            gdf[cols_names[i]] = gdf[cols_names[i]].astype('int32')
        gdf.drop(['name_neighbor', 'geometry', 'key_borough', 'name_borough'], axis=1, inplace=True)
        print(gdf.isna().sum())
        return gdf
    
    def countLinesIntersectPolygon(self, list_shapefiles, polygon_shapefile):
        cols_names = ['cablebus_lines', 'ptransp_routes', 'main_roads', 'metro_lines', 'rtp_lines', 'train_lines', 'trolebus_lines']
        gdf = polygon_shapefile.copy()
        for i in range(0, len(list_shapefiles)):
            gdf = gdf.merge(gdf.sjoin(list_shapefiles[i], predicate='intersects').groupby('key_neighbor').size().rename(cols_names[i]).reset_index(), how='left').fillna(0)
            gdf[cols_names[i]] = gdf[cols_names[i]].astype('int32')
        gdf.drop(['name_neighbor', 'geometry', 'key_borough', 'name_borough'], axis=1, inplace=True)
        return gdf
    
    def mergeAttributesPolygons(self, list_gdfs, polygon_shapefile, type_merge):
        gdfs = polygon_shapefile.copy()
        for gdf in list_gdfs:
            gdfs = gdfs.merge(gdf, how="left", on="key_neighbor")
        if type_merge == 'aggregated':
            gdfs.drop(['name_neighbor', 'geometry', 'key_borough', 'name_borough'], axis=1, inplace=True)
            for col in ['centres_vaw', 'be_schools', 'commercial_units', 'industrial_units', 'service_units']:
                gdfs[col] = gdfs[col].fillna(0)
                gdfs[col] = gdfs[col].astype('int32')
        elif type_merge == 'final':
            pass
        else:
            print('Wrong input.')
        print(gdfs.isna().sum(), '\n')
        print("Geodataframe shape: ", gdfs.shape)
        return gdfs
    
    def setGeometryProjection(self, data):
        def createDataGeometry(data_in):
            gdf = gp.GeoDataFrame(data_in, geometry=gp.points_from_xy(data_in.longitude, data_in.latitude))
            return gdf
        def setProjection(data_in):
            gdf = data_in.set_crs("EPSG:4326")
            gdf = gdf.to_crs("EPSG:32614")
            print("CRS projection of data: " + str(gdf.crs) + '\n')
            return gdf
        gdf_final = createDataGeometry(data)
        gdf_final = setProjection(gdf_final)
        return gdf_final
    
    def joinPointsAttributes(self, list_shapefiles, polygon_shapefile):
        merged_shapefiles = []
        for shapefile in list_shapefiles:
            gdf = polygon_shapefile.sjoin(shapefile, predicate='contains', how='left')
            gdf.drop(['name_neigh', 'key_boroug', 'name_borou', 'cablebus_s', 'commer_ven', 'health_cen', 'metro_sta', 'pmarkets', 'pparking', 'hospitals', 'train_sta', 'trolebus_s', 'cablebus_l', 'ptransp_ro', 'main_roads', 'metro_line', 'rtp_lines', 'train_line', 'trolebus_l', 'centres_va', 'be_schools', 'commercial', 'industrial', 'service_un', 'geometry', 'index_right'], axis=1, inplace=True)
            merged_shapefiles.append(gdf)
        return merged_shapefiles[0], merged_shapefiles[1]
    
    def saveToShapefile(self, gdf):
        gdf.to_file(self.proc_data_directory + '/shapefiles/preprocessed_shapefiles_data.shp', driver ='ESRI Shapefile')
        
    def saveToCSV(self, gdf):
        gdf.to_csv(self.proc_data_directory + '/preprocessed_shapefiles_data.csv', index=False)
        
    def loadProcessedData(self):
        proc_data_directory = os.path.normpath(os.path.join(self.current_directory, '../data/preprocessed'))
        path = proc_data_directory + '/shapefiles/'
        shapefile = glob.glob(os.path.join(path, "*.shp"))
        gdf = gp.read_file(shapefile[0])
        print("Shapefile read: " + str(shapefile[0]) + '\n')
        return gdf
    
        