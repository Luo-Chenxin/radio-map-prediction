from pathlib import Path
import csv

def get_radiounet_model(config_data, model_class):
    in_channels = config_data.get_in_channels()
    first_out_channels = 6 if in_channels <= 3 else 10
    return model_class(in_channels, first_out_channels)

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