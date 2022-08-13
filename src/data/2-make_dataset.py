import pyhere
import sys
sys.path.insert(0, str(pyhere.here().resolve().joinpath("src")))
import utils

import click
import logging

# from dotenv import find_dotenv, load_dotenv
import os, json, requests
from io import StringIO
import certifi
import urllib3
import numpy as np
import pandas as pd
import pyhere
from time import sleep

@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """
    1. Fetch climatic factors of the POWER NASA API given a Latitude and Longitude and a range of years.
    2. The data fetched comes divided in months, so it is grouped by seasons to be consistent between regions from North and South hemispheres.
    3. The data is concatenated with some parameters of the Global Power Database data previously obtained from World Resources Institute (and previously manipulated too).
    4. The concatenation is done where the indexes match.
    5. In each chunk iteration overwrites a final csv file that at the end of the process results in a final csv to start working for creating models. 
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    

    csv_power_plants = pd.read_csv(utils.DIR_DATA_INTERIM/"power_plants_with_generation_transformed.csv", index_col=[0])
    max_index_csv_power_plants = len(csv_power_plants.index)

    # POWER NASA PARAMETERS THAT WERE CONSIDERED TO FETCH:

    # TQV                   MERRA-2 Total Column Precipitable Water (kg m-2) 
    # WS10M                 MERRA-2 Wind Speed at 10 Meters (m/s) 
    # CLRSKY_SFC_SW_DNI     CERES SYN1deg Clear Sky Surface Shortwave Downward Direct Normal Irradiance (kW-hr/m^2/day) 
    
    # ALLSKY_SFC_SW_DWN     All Sky Surface Shortwave Downward Irradiance
    # CLRSKY_SFC_SW_DWN     Clear Sky Surface Shortwave Downward Irradiance
    # x ALLSKY_KT             All Sky Insolation Clearness Index
    # WS10M_MIN_AVG         Wind Speed at 10 Meters Minimum Average
    # WS10M_MAX_AVG         Wind Speed at 10 Meters Maximum Average
    # WS50M_MAX_AVG         Wind Speed at 50 Meters Maximum Average
    # WS50M_MIN_AVG         Wind Speed at 50 Meters Minimum Average
    
    # ALLSKY_SFC_SW_DIFF    All Sky Surface Shortwave Diffuse Irradiance. The diffuse (light energy scattered out of the direction of the sun) solar irradiance incident on a horizontal plane at the surface of the earth under all sky conditions. (kW-hr/m^2/day)
    # ALLSKY_SFC_SW_UP      The upward shortwave irradiance under all sky conditions. (kW-hr/m^2/day)
    # ALLSKY_SFC_SW_UP_SD   All Sky Surface Shortwave Upward Irradiance Standard Deviation (kW-hr/m^2/day)
    # ALLSKY_SFC_SW_DIFF_SD All Sky Surface Shortwave Diffuse Irradiance Standard Deviation
    # ALLSKY_SFC_LW_DWN     All Sky Surface Longwave Downward Irradiance
    # ALLSKY_SFC_LW_UP      All Sky Surface Longwave Upward Irradiance (W/m^2)
    
    # ALLSKY_SFC_SW_DNI     All Sky Surface Shortwave Downward Direct Normal Irradiance
    # ALLSKY_SFC_SW_UP_SD   All Sky Surface Shortwave Upward Irradiance Standard Deviation
    # WS50M                 Wind Speed at 50 Meters
    # WS50M_RANGE_AVG       Wind Speed at 50 Meters Range Average
    # WS10M_RANGE_AVG       Wind Speed at 10 Meters Range Average
    # WS10M                 Wind Speed at 10 Meters
    # ALLSKY_SFC_SW_DNI_MAX_RD  All Sky Surface Shortwave Downward Direct Normal Irradiance Maximum Relative Difference
    # ALLSKY_SFC_SW_UP_MAX  All Sky Surface Shortwave Upward Irradiance Maximum
    # CLRSKY_SFC_SW_DIFF    Clear Sky Surface Shortwave Downward Diffuse Horizontal Irradiance
    # CLRSKY_SFC_SW_DNI     Clear Sky Surface Shortwave Downward Direct Normal Irradiance
    # CLRSKY_SFC_SW_UP      Clear Sky Surface Shortwave Upward Irradiance

    # T2M                   Temperature at 2 Meters
    
    # CLOUD_AMT_DAY         The average percent of cloud amount during daylight.
    
    
    index_reference = 0
    rows_chunk = 60
    seconds_to_sleep = 2
    url_parameters = ["ALLSKY_SFC_SW_DWN",
                        "CLRSKY_SFC_SW_DWN",
                        # "ALLSKY_SFC_SW_DIFF",
                        "ALLSKY_SFC_SW_UP",
                        # "ALLSKY_SFC_LW_DWN",
                        # "ALLSKY_SFC_LW_UP",
                        "ALLSKY_SFC_SW_DNI",
                        # "ALLSKY_SFC_SW_DNI_MAX_RD",
                        "ALLSKY_SFC_SW_UP_MAX",
                        # "CLRSKY_SFC_SW_DIFF",
                        "CLRSKY_SFC_SW_DNI",
                        "CLRSKY_SFC_SW_UP",
                        #"ALLSKY_KT",
                        "CLOUD_AMT_DAY",
                        "WS10M_MAX_AVG",
                        "WS50M_MAX_AVG",
                        "WS50M",
                        "WS50M_RANGE_AVG",
                        "WS10M",
                        "WS10M_RANGE_AVG",
                        "T2M"
                    ]
    ORDINAL_COLUMNS =   [
                '1st(min)',
                '2nd',
                '3rd',
                '4th',
                '5th',
                '6th',
                '7th',
                '8th',
                '9th',
                '10th',
                '11th',
                '12th(max)'
            ]
    while index_reference + rows_chunk <= (max_index_csv_power_plants - 1):
        try:
            df_transformed_data_combined_with_nasa = pd.read_csv(utils.DIR_DATA_EXTERNAL/"v7_transformed_data_combined_with_nasa.csv", index_col=['index'] )
            index_reference = df_transformed_data_combined_with_nasa.index.max() + 1
        except FileNotFoundError: 
            pass
        
        top_to_iterate = None
    
        if (index_reference + (rows_chunk - 1)) <= max_index_csv_power_plants:
            top_to_iterate = index_reference + (rows_chunk - 1)
        else:
            top_to_iterate = max_index_csv_power_plants -1
        sample_lat_lon = csv_power_plants.loc[index_reference:top_to_iterate, ["latitude", "longitude"]]
        print(f'{index_reference}:{top_to_iterate} de {max_index_csv_power_plants - 1}')
    
    
    
        locations = list(sample_lat_lon.to_records(index=False))
    
       
    
        output = r""
        base_url = r"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={url_parameters}&community=RE&longitude={longitude}&latitude={latitude}&start=2013&end=2019&format=CSV&header=false"
        df_response = pd.DataFrame()
    
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            ca_certs=certifi.where()
        )
    
        aux_counter_index = index_reference
    
        filename_template = "v7_transformed_data_combined_with_nasa.csv"
        filename = filename_template
        for latitude, longitude in locations:
            api_request_url = base_url.format(longitude=longitude, latitude=latitude, url_parameters=','.join(url_parameters))
            response = http.request('GET', api_request_url, timeout=30.00, retries=urllib3.util.Retry(total=5, backoff_factor=1, status_forcelist=[504]))
            
            response_data = response.data.decode('utf-8')
            df_response_aux = pd.read_csv(StringIO(response_data))
            df_response_aux["latitude"] = latitude
            df_response_aux["longitude"] = longitude
            # print(df_response_aux)
            # if longitude > 0:
            #     hemisphere_months_seasons = utils.NORTH_HEMISPHERE_MONTHS_SEASONS
            # else:
            #     hemisphere_months_seasons = utils.SOUTH_HEMISPHERE_MONTHS_SEASONS
            # for index, element in hemisphere_months_seasons.items():
            #     df_response_aux[index]= df_response_aux[element].mean(axis=1)
    
            # df_response_aux.drop(columns= utils.MONTHS_OF_YEAR, inplace = True)

            series_sorted_values_by_column = df_response_aux[utils.MONTHS_OF_YEAR].apply(lambda row: row.sort_values().values, axis=1)
            df_aux_statistics = pd.DataFrame(series_sorted_values_by_column.values.tolist(), columns=ORDINAL_COLUMNS)
            df_aux_statistics['mean'] = np.around(df_aux_statistics[ORDINAL_COLUMNS].mean(axis=1),3)
            df_aux_statistics['std'] = np.around(df_aux_statistics[ORDINAL_COLUMNS].std(axis=1),3)
            df_aux_statistics['median'] = (df_aux_statistics['6th'] + df_aux_statistics['7th']) / 2
            df_aux_statistics['min'] = df_aux_statistics['1st(min)']
            df_aux_statistics['max'] = df_aux_statistics['12th(max)']
            df_aux_statistics.drop(columns= ORDINAL_COLUMNS, inplace = True)

            df_response_aux = pd.concat([df_response_aux, df_aux_statistics], axis=1)
            df_response_aux.drop(columns= utils.MONTHS_OF_YEAR, inplace = True)
            # "PIVAT! PIVAT! PIVAT!"
            df_response_aux = df_response_aux.pivot_table(index=["latitude", "longitude"], columns=["PARAMETER", "YEAR"])
            df_response_aux.columns = ["_".join(map(str, cols)) for cols in df_response_aux.columns.to_flat_index()]
    
            pd.concat([df_response_aux, csv_power_plants.loc[aux_counter_index, ['capacity_mw', 'primary_fuel_transformed']]], axis=1)
    
            if(df_response.empty):
            
                df_response = df_response_aux.copy()
            else:
                df_response = pd.concat([df_response,df_response_aux])
    
            aux_counter_index += 1
    
    
        df_response.reset_index(inplace = True)
        df_response.index += index_reference
        # TRANSFORMING AND COMBINING DATA
        try:
        
            df_response = pd.concat([df_response,df_transformed_data_combined_with_nasa])
            df_response.sort_index(inplace = True)
            del df_transformed_data_combined_with_nasa
        
        except NameError:
            pass
        
        df_response.to_csv(utils.DIR_DATA_EXTERNAL/filename, index_label='index')
        del df_response
        del df_response_aux
        
        logger.info('Sleeping...')
        sleep(seconds_to_sleep)
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
