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

_NUMBER_OF_WORKER_PROCESSES = max(1, multiprocessing.cpu_count() - 1)

_KEY_ARRAY = "array"
_KEY_SHAPE = "shape"

_global_shared_arrays = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


# --- Core Annual SPEI Computation ---

def spei_annual_loglogistic(
    series_d: np.ndarray,
    start_year: int,
    ref_start: int,
    ref_end: int
) -> np.ndarray:

    years = np.arange(start_year, start_year + len(series_d))
    d_series = pd.Series(series_d, index=years)

    data_ref = d_series.loc[(d_series.index >= ref_start) & (d_series.index <= ref_end)]
    data_ref_valid = data_ref.dropna()
    data_all = d_series

    if len(data_ref_valid) < 15:
        return np.full(len(series_d), np.nan)

    spei_values = np.full(len(series_d), np.nan)

    try:
        min_val = np.nanmin(data_all.values)
        if np.isnan(min_val):
            return spei_values

        xi = min_val - 0.001

        values_fit = data_ref_valid.values - xi
        values_calc = data_all.values - xi

        shape, loc, fit_scale = stats.fisk.fit(values_fit, floc=0)
        probabilities = stats.fisk.cdf(values_calc, c=shape, loc=loc, scale=fit_scale)
        probabilities = np.where(np.isnan(probabilities), np.nan, np.clip(probabilities, 1e-6, 1 - 1e-6))
        spei_values = stats.norm.ppf(probabilities)

    except Exception:
        pass

    return spei_values


def _spei_worker_function(precips: np.ndarray, pets: np.ndarray, parameters: Dict) -> np.ndarray:
    diff = precips - pets
    if np.all(np.isnan(diff)):
        return np.full(diff.shape, np.nan)
    return spei_annual_loglogistic(
        diff,
        start_year=parameters['data_start_year'],
        ref_start=parameters['ref_start'],
        ref_end=parameters['ref_end']
    )


# --- Parallel Processing Framework ---

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
    model_name = params.get('model_name', 'Unknown')

    try:
        ds_precip = xr.open_dataset(params['netcdf_precip'])
        ds_pet = xr.open_dataset(params['netcdf_pet'])
    except FileNotFoundError as e:
        _logger.error(f"[{model_name}] File error: {e}")
        return

    try:
        if 'lat' in ds_precip.dims and 'lon' in ds_precip.dims:
            ds_precip = ds_precip.transpose('lat', 'lon', 'time')
            ds_pet = ds_pet.transpose('lat', 'lon', 'time')
    except ValueError:
        _logger.error(f"[{model_name}] Dimension transposition failed.")
        return

    precip_values = ds_precip[params['var_name_precip']].values
    pet_values = ds_pet[params['var_name_pet']].values

    if precip_values.shape != pet_values.shape:
        _logger.error(f"[{model_name}] Shape mismatch: PR {precip_values.shape} vs PET {pet_values.shape}")
        return

    output_shape = precip_values.shape

    try:
        time_vals = ds_precip['time'].values
        if len(time_vals) > 0:
            first_time = time_vals[0]
            if hasattr(first_time, 'year'):
                data_start_year = first_time.year
            else:
                data_start_year = pd.to_datetime(first_time).year
        else:
            raise ValueError("Time dimension empty")
        _logger.info(f"[{model_name}] Data start year: {data_start_year}, total years: {len(time_vals)}")
    except Exception as e:
        _logger.error(f"[{model_name}] Time parsing failed, defaulting to 1900: {e}")
        data_start_year = 1900

    if params['ref_start'] < data_start_year:
        _logger.warning(
            f"[{model_name}] Warning: Reference period start ({params['ref_start']}) "
            f"is earlier than data start ({data_start_year})."
        )

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

    calc_args = {
        'data_start_year': data_start_year,
        'ref_start': params['ref_start'],
        'ref_end': params['ref_end']
    }

    input_vars = {'var_name_precip': params['var_name_precip'], 'var_name_pet': params['var_name_pet']}

    _logger.info(
        f"[{model_name}] Starting parallel computation "
        f"(Annual SPEI, reference period: {params['ref_start']}-{params['ref_end']})..."
    )
    parallel_process(_global_shared_arrays, input_vars, internal_var_name, calc_args)

    result_array = np.frombuffer(
        _global_shared_arrays[internal_var_name][_KEY_ARRAY].get_obj()
    ).reshape(output_shape)

    output_ds = xr.Dataset(
        {internal_var_name: (ds_precip[params['var_name_precip']].dims, result_array)},
        coords=ds_precip.coords
    )

    output_ds[internal_var_name].attrs['long_name'] = f'Annual SPEI ({model_name})'
    output_ds[internal_var_name].attrs['units'] = '1'
    output_ds[internal_var_name].attrs['comment'] = 'Calculated from Annual P and Annual PET'
    output_ds[internal_var_name].attrs['reference_period'] = f"{params['ref_start']}-{params['ref_end']}"

    output_file_path = os.path.join(params['output_dir'], params['output_filename'])
    output_ds.to_netcdf(output_file_path)
    _logger.info(f"[{model_name}] Done! Results saved to: {params['output_filename']}")


# --- Batch Processing ---

def batch_process_models():
    # ===== Modify the paths below to match your directory structure =====
    pr_dir = "/home/jijiachen/pr_annual"     # Directory containing annual precipitation NetCDF files
    pet_dir = "/home/jijiachen/pet_annual"   # Directory containing annual PET NetCDF files
    output_dir = "/home/jijiachen/spei_annual"    # Output directory for SPEI results

    os.makedirs(output_dir, exist_ok=True)

    # ===== Modify the variable names to match those in your NetCDF files =====
    VAR_PR = "pre"       # Precipitation variable name in the NetCDF files
    VAR_PET = "pet"      # PET variable name in the NetCDF files

    # ===== Modify the reference period as needed =====
    REF_START = 1961
    REF_END = 2024

    # ===== Optionally set a priority model to be processed first =====
    PRIORITY_MODEL = "EC-Earth3-Veg"

    _logger.info("Scanning file directories...")

    pet_map = {}
    if os.path.exists(pet_dir):
        for f in os.listdir(pet_dir):
            if f.endswith(".nc"):
                try:
                    parts = f.split('_')
                    if len(parts) > 1:
                        # ===== Adjust the index below based on your PET filename format =====
                        # e.g., for "pet_ACCESS-CM2_ssp585_annual.nc", parts[1] = "ACCESS-CM2"
                        model_name = parts[1]
                        pet_map[model_name] = os.path.join(pet_dir, f)
                except IndexError:
                    pass
    else:
        _logger.error(f"PET directory does not exist: {pet_dir}")
        return

    if not os.path.exists(pr_dir):
        _logger.error(f"PR directory does not exist: {pr_dir}")
        return

    pr_files = [f for f in os.listdir(pr_dir) if f.endswith(".nc")]
    pr_files.sort(key=lambda x: 0 if PRIORITY_MODEL in x else 1)

    total_files = len(pr_files)
    _logger.info(f"Found {total_files} PR files. Built {len(pet_map)} PET mappings.")

    sorted_model_names = sorted(pet_map.keys(), key=len, reverse=True)

    count = 0
    for pr_file in pr_files:
        try:
            matched_model = None
            for known_model in sorted_model_names:
                if known_model in pr_file:
                    matched_model = known_model
                    break

            if matched_model:
                count += 1
                pr_path = os.path.join(pr_dir, pr_file)
                pet_path = pet_map[matched_model]

                output_filename = f"{matched_model}_spei_annual.nc"

                if os.path.exists(os.path.join(output_dir, output_filename)):
                    _logger.info(f"[{count}] {matched_model} already exists, skipping.")
                    continue

                _logger.info(f"\n{'='*10} Processing model [{count}]: {matched_model} {'='*10}")

                params = {
                    "netcdf_precip": pr_path,
                    "var_name_precip": VAR_PR,
                    "netcdf_pet": pet_path,
                    "var_name_pet": VAR_PET,
                    "output_dir": output_dir,
                    "output_filename": output_filename,
                    "model_name": matched_model,
                    "ref_start": REF_START,
                    "ref_end": REF_END
                }

                compute_and_write_spei(params)

            else:
                _logger.warning(f"No matching PET file found for PR file: {pr_file}")

        except Exception as e:
            _logger.error(f"Unknown error while processing {pr_file}: {e}")

    _logger.info("\nAll matched models processed!")


if __name__ == "__main__":
    batch_process_models()