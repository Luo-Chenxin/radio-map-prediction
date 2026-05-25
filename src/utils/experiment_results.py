from pathlib import Path
import csv
from datetime import datetime

def get_dataset_field(config) -> str:
    """
    Get dataset field for append_to_csv function
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


def append_record(file_path, model_arch, train_set, test_set, metrics):
    """
    Add a record to the CSV file
    
    Parameter:
      file_path: str, path to the CSV file
      model_arch: str, model architecture name
      train_set: str, training set field.
      test_set: str, testing set field.
      metrics: dict, test results (include MSE, NMSE, Time_Per_Sample_Sec)
    """
    
    # Define headers
    headers = [
        'Model_Architecture', 
        'Train_Set_Desc', 
        'Test_Set_Desc', 
        'MSE', 
        'NMSE', 
        'Time_Per_Sample_Sec', 
        'Recording_Time'
    ]
    
    recording_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    row_data = [
        model_arch,
        train_set,
        test_set,
        metrics['MSE'],
        metrics['NMSE'],
        metrics['Time_Per_Sample_Sec'],
        recording_time
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