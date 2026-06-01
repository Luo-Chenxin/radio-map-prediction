from pathlib import Path
import csv

def get_radiounet_model(config, model_class):
    in_channels = 1 + 1
    if config.samples_number > 0:
        in_channels = in_channels + 1
    if config.cars_exist:
        in_channels = in_channels + 1
    
    first_out_channels = 6 if in_channels <= 3 else 10
    return model_class(in_channels, first_out_channels)


def get_dataset_desc(config) -> str:
    """
    Get dataset description for append_record function
    """
    simulationStr = config.simulation if config.sparse_IRT4_number == 0 else f"IRT4_Adapter_{config.sparse_IRT4_number}" 
    carsStr = "Cars_Exist" if config.cars_exist else "No_Cars_Exist"
    if config.city_map == 'complete':
        cityMapStr = 'Complete_City_Map'
    elif config.city_map == 'missing':
        cityMapStr = f'City_Map_With_{config.missing}_Missing_Buildings'
    elif config.city_map == 'rand':
        cityMapStr = 'City_Map_With_Random_Missing_Buildings'
    else: 
        cityMapStr = f'Unknown_{config.city_map}'
    samplesStr = f"Input_Samples_{config.samples_number}"

    return f"{simulationStr}|{carsStr}|{cityMapStr}|{samplesStr}"


def append_record(file_path, model_arch, dataset_desc, metrics, timestamp):
    """
    Add a record to the CSV file, unified with TensorBoard timestamp.
    
    Parameter:
      file_path: str, path to the CSV file
      model_arch: str, model architecture name
      dataset_desc: str, dataset description field
      metrics: dict, test results (include RMSE, NMSE, Time_Per_Sample_Sec)
      timestamp: str, the exact timestamp
    """
    
    # Define headers
    headers = [
        'Model_Architecture',
        'Dataset_Desc',
        'RMSE', 
        'NMSE', 
        'Time_Per_Sample_Sec', 
        'Recording_Time'
    ]

    # Match the row data with headers precisely
    row_data = [
        model_arch,
        dataset_desc,
        metrics['RMSE'],
        metrics['NMSE'],
        metrics['Time_Per_Sample_Sec'],
        timestamp
    ]

    file_path_obj = Path(file_path)
    file_exists = file_path_obj.exists()
    
    with file_path_obj.open(mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # If the file does not exist, write the header first.
        if not file_exists:
            writer.writerow(headers)
            
        # Write the actual data row
        writer.writerow(row_data)