import logging
import multiprocessing
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# --- Global Configuration ---

# Set number of parallel worker processes
_NUMBER_OF_WORKER_PROCESSES = max(1, multiprocessing.cpu_count() - 1)

# Shared memory dictionary keys
_KEY_ARRAY = "array"
_KEY_SHAPE = "shape"

_global_shared_arrays = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


# --- Core SPEI Calculation Logic ---

def spei_loglogistic3(
    series_d: np.ndarray,
    scale: int,
    start_year: int,
    ref_start: int,
    ref_end: int,
    periodicity: str = "monthly"
) -> np.ndarray:
    """
    Calculate SPEI (reference period approach).
    1. Use the full series to establish offset (xi) to prevent negative values.
    2. Fit parameters using only data from [ref_start, ref_end] period.
    3. Apply parameters to the full series to calculate CDF and SPEI.
    """
    if periodicity != "monthly":
        raise NotImplementedError("Only 'monthly' periodicity is supported.")

    # Build complete time index
    dates = pd.date_range(start=f'{start_year}-01-01', periods=len(series_d), freq='MS')
    d_series = pd.Series(series_d, index=dates)
    
    # Calculate rolling accumulation
    d_accum = d_series.rolling(window=scale, min_periods=scale).sum().dropna()

    if d_accum.empty:
        return np.full(len(series_d), np.nan)

    spei_results = pd.Series(np.nan, index=d_series.index)

    for month in range(1, 13):
        # Get all data for this month (historical + future)
        month_data_all = d_accum[d_accum.index.month == month]
        
        # Get reference period data for fitting (1961-2024)
        month_data_ref = month_data_all[
            (month_data_all.index.year >= ref_start) & 
            (month_data_all.index.year <= ref_end)
        ]

        # If reference period data is too small for fitting, skip this month entirely
        if len(month_data_ref) < 15:  
            continue

        try:
            # 1. Determine offset xi
            # Must use the minimum value from the [full series] to determine xi,
            # otherwise if future conditions are drier than historical,
            # subtracting historical xi will produce negative values,
            # causing log-logistic distribution errors.
            xi = month_data_all.min() - 0.001
            
            # Prepare data:
            # data_fit: for training parameters (reference period)
            # data_calc: for calculating SPEI (full period)
            data_fit = month_data_ref.values - xi
            data_calc = month_data_all.values - xi

            # 2. Fit (using only reference period data)
            shape, loc, fit_scale = stats.fisk.fit(data_fit, floc=0)

            # 3. Calculate probabilities (apply to full series)
            probabilities = stats.fisk.cdf(data_calc, c=shape, loc=loc, scale=fit_scale)

            # Clip probability range to prevent Inf
            probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)

            # 4. Convert to Z-score
            spei_values = stats.norm.ppf(probabilities)
            
            # Fill back results
            spei_results.loc[month_data_all.index] = spei_values

        except Exception as e:
            continue

    return spei_results.values


def _spei_worker_function(precips: np.ndarray, pets: np.ndarray, parameters: Dict) -> np.ndarray:
    diff = precips - pets
    return spei_loglogistic3(
        diff,
        scale=parameters['scale'],
        start_year=parameters['data_start_year'],
        ref_start=parameters['ref_start'],
        ref_end=parameters['ref_end'],
        periodicity=parameters['periodicity']
    )


# --- Parallel Processing Framework (unchanged) ---

def init_worker(shared_arrays_dict: Dict):
    global _global_shared_arrays
    _global_shared_arrays = shared_arrays_dict


def _apply_along_axis_double(params: Dict[str, Any]):
    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]
    
    shape = _global_shared_arrays[params["output_var_name"]][_KEY_SHAPE]
    
    precip_array = np.frombuffer(
        _global_shared_arrays[params["var_name_precip"]][_KEY_ARRAY].get_obj()
    ).reshape(shape)
    
    pet_array = np.frombuffer(
        _global_shared_arrays[params["var_name_pet"]][_KEY_ARRAY].get_obj()
    ).reshape(shape)
    
    sub_array_precip = precip_array[start_index:end_index]
    sub_array_pet = pet_array[start_index:end_index]

    output_array_buffer = _global_shared_arrays[params["output_var_name"]][_KEY_ARRAY]
    computed_array_slice = np.frombuffer(output_array_buffer.get_obj()).reshape(shape)[start_index:end_index]

    for i in range(sub_array_precip.shape[0]):
        for j in range(sub_array_precip.shape[1]):
            precip_timeseries = sub_array_precip[i, j, :]
            pet_timeseries = sub_array_pet[i, j, :]
            
            if np.all(np.isnan(precip_timeseries)) or np.all(np.isnan(pet_timeseries)):
                computed_array_slice[i, j, :] = np.nan
                continue

            result_timeseries = func1d(precip_timeseries, pet_timeseries, parameters=params["args"])
            computed_array_slice[i, j, :] = result_timeseries


def parallel_process(arrays_dict: Dict, input_var_names: Dict, output_var_name: str, args: Dict):
    shape = arrays_dict[output_var_name][_KEY_SHAPE]
    num_processes = min(shape[0], _NUMBER_OF_WORKER_PROCESSES)
    if num_processes == 0:
        return
        
    indices = np.array_split(range(shape[0]), num_processes)
    
    chunk_params = []
    for i in range(num_processes):
        params = {
            "func1d": _spei_worker_function,
            "var_name_precip": input_var_names["var_name_precip"],
            "var_name_pet": input_var_names["var_name_pet"],
            "output_var_name": output_var_name,
            "sub_array_start": indices[i][0],
            "sub_array_end": indices[i][-1] + 1,
            "args": args,
        }
        chunk_params.append(params)

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(arrays_dict,),
    ) as pool:
        pool.map(_apply_along_axis_double, chunk_params)


def compute_and_write_spei(params: Dict):
    scale = params['scale']
    model_name = params.get('model_name', 'Unknown')
    
    # Read data
    try:
        ds_precip = xr.open_dataset(params['netcdf_precip'])
        ds_pet = xr.open_dataset(params['netcdf_pet'])
    except FileNotFoundError as e:
        _logger.error(f"[{model_name}] File error: {e}")
        return

    try:
        ds_precip = ds_precip.transpose('lat', 'lon', 'time')
        ds_pet = ds_pet.transpose('lat', 'lon', 'time')
    except ValueError:
        _logger.error(f"[{model_name}] Dimension transpose failed")
        return

    precip_values = ds_precip[params['var_name_precip']].values
    pet_values = ds_pet[params['var_name_pet']].values
    output_shape = precip_values.shape
    
    # Get data start year
    try:
        time_val = ds_precip['time'].values[0]
        if hasattr(time_val, 'year'):
             data_start_year = time_val.year
        else:
             data_start_year = pd.to_datetime(time_val).year
        _logger.info(f"[{model_name}] Data start year: {data_start_year}")
    except Exception as e:
        _logger.error(f"[{model_name}] Time parsing failed, defaulting to 1900: {e}")
        data_start_year = 1900

    # Validate if reference period is within data range
    if params['ref_start'] < data_start_year:
        _logger.warning(f"[{model_name}] Warning: Specified reference period start year ({params['ref_start']}) is earlier than data start year ({data_start_year}), which may cause calculation errors.")

    # Shared memory setup
    _global_shared_arrays.clear()
    _global_shared_arrays[params['var_name_precip']] = {
        _KEY_ARRAY: multiprocessing.Array("d", precip_values.flatten()),
        _KEY_SHAPE: output_shape,
    }
    _global_shared_arrays[params['var_name_pet']] = {
        _KEY_ARRAY: multiprocessing.Array("d", pet_values.flatten()),
        _KEY_SHAPE: output_shape,
    }
    
    internal_var_name = "spei"
    _global_shared_arrays[internal_var_name] = {
        _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
        _KEY_SHAPE: output_shape,
    }

    # --- Parameter passing update ---
    calc_args = {
        'scale': scale, 
        'data_start_year': data_start_year, 
        'periodicity': 'monthly',
        'ref_start': params['ref_start'], 
        'ref_end': params['ref_end']      
    }
    
    input_vars = {'var_name_precip': params['var_name_precip'], 'var_name_pet': params['var_name_pet']}
    
    _logger.info(f"[{model_name}] Starting parallel computation (reference period: {params['ref_start']}-{params['ref_end']})...")
    parallel_process(_global_shared_arrays, input_vars, internal_var_name, calc_args)

    # Write results
    result_array = np.frombuffer(_global_shared_arrays[internal_var_name][_KEY_ARRAY].get_obj()).reshape(output_shape)
    
    output_ds = xr.Dataset(
        {internal_var_name: (ds_precip[params['var_name_precip']].dims, result_array)},
        coords={'lat': ds_precip['lat'], 'lon': ds_precip['lon'], 'time': ds_precip['time']}
    )
    
    output_ds[internal_var_name].attrs['long_name'] = f'SPEI-{scale} ({model_name})'
    output_ds[internal_var_name].attrs['units'] = '1'
    output_ds[internal_var_name].attrs['reference_period'] = f"{params['ref_start']}-{params['ref_end']}"
    
    output_file_path = os.path.join(params['output_dir'], params['output_filename'])
    output_ds.to_netcdf(output_file_path)
    _logger.info(f"[{model_name}] Complete! Results saved: {params['output_filename']}")


# --- Batch Processing Main Logic ---

def batch_process_models():
    # 1. Define paths
    pr_dir = "/home/jijiachen/data_jjc/pr/ssp245"
    pet_dir = "/home/jijiachen/data_jjc/pet/ssp245"
    output_dir = "/home/jijiachen/data_jjc/spei_ssp245"
    
    os.makedirs(output_dir, exist_ok=True)

    VAR_PR = "pre"     
    VAR_PET = "pet"    
    SCALE = 3
    
    # --- Reference period settings ---
    REF_START = 1961
    REF_END = 2024

    # Priority model keyword
    PRIORITY_MODEL = "ACCESS-CM2"

    _logger.info("Scanning file directories...")
    
    pet_map = {}
    for f in os.listdir(pet_dir):
        if f.endswith(".nc"):
            try:
                model_name = f.split('_')[1] 
                pet_map[model_name] = os.path.join(pet_dir, f)
            except IndexError:
                pass

    pr_files = [f for f in os.listdir(pr_dir) if f.endswith(".nc")]
    
    # --- Priority sorting logic ---
    pr_files.sort(key=lambda x: 0 if PRIORITY_MODEL in x else 1)
    
    total_files = len(pr_files)
    _logger.info(f"Found {total_files} files. Priority: {PRIORITY_MODEL} will be processed first.")

    count = 0
    for pr_file in pr_files:
        try:
            parts = pr_file.split('_')
            if len(parts) < 3:
                continue
            
            model_name = parts[2]
            
            if model_name in pet_map:
                count += 1
                pet_path = pet_map[model_name]
                pr_path = os.path.join(pr_dir, pr_file)
                
                output_filename = f"{model_name}_spei{SCALE}.nc"
                
                if os.path.exists(os.path.join(output_dir, output_filename)):
                    _logger.info(f"[{count}/{len(pet_map)}] {model_name} already exists, skipping.")
                    continue

                _logger.info(f"\n{'='*10} Processing model [{count}]: {model_name} {'='*10}")
                
                params = {
                    "netcdf_precip": pr_path,
                    "var_name_precip": VAR_PR,
                    "netcdf_pet": pet_path,
                    "var_name_pet": VAR_PET,
                    "scale": SCALE,
                    "output_dir": output_dir,
                    "output_filename": output_filename,
                    "model_name": model_name,
                    "ref_start": REF_START,
                    "ref_end": REF_END
                }
                
                compute_and_write_spei(params)
                
            else:
                _logger.warning(f"PET file not found for model {model_name}")
                
        except Exception as e:
            _logger.error(f"Unknown error occurred while processing file {pr_file}: {e}")

    _logger.info("\nAll matching models have been processed!")


if __name__ == "__main__":
    batch_process_models()