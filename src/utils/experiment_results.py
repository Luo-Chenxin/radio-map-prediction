from pathlib import Path
import csv
from datetime import datetime

def get_dataset_field(
        carsInput:bool, 
        carsSimulation:bool, 
        missing:bool, 
        samples:bool, 
        irt4_adapter:bool) -> str:
    """
    get train_set field or test_set field for append_to_csv function
    Parameter:
      carsInput: bool, input cars or not
      carsSimulation: bool, use cars in simulation or not
      missing: bool, some buildings are missing or not
      samples: bool, input samples or not
      irt4_adapter: bool, use some IRT4 simulation targets to adapt or not.
    """
    carsInputStr = "CarsIn" if carsInput else "NoCarsIn"
    carsSimulationStr = "CarsSim" if carsSimulation else "NoCarsSim"
    missingStr = "Missing" if missing else "Complete"
    samplesStr = "Samples" if samples else "Clean"
    irt4_adapterStr = "IRT4" if irt4_adapter else "NoIRT4"

    return f"{carsInputStr}|{carsSimulationStr}|{missingStr}|{samplesStr}|{irt4_adapterStr}"


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