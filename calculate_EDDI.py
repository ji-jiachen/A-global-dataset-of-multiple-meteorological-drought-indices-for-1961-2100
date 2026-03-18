import logging
import multiprocessing
import os
from datetime import datetime
from typing import Any, Dict, List

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


# --- EDDI Computation Logic ---

def eddi_loglogistic(
    series_pet: np.ndarray,
    scale: int,
    start_year: int,
    ref_start: int,
    ref_end: int,
    periodicity: str = "monthly"
) -> np.ndarray:

    if periodicity != "monthly":
        raise NotImplementedError("Only 'monthly' periodicity is supported.")

    dates = pd.date_range(start=f'{start_year}-01-01', periods=len(series_pet), freq='MS')
    pet_series = pd.Series(series_pet, index=dates)

    # Accumulate PET
    pet_accum = pet_series.rolling(window=scale, min_periods=scale).sum().dropna()

    if pet_accum.empty:
        return np.full(len(series_pet), np.nan)

    eddi_results = pd.Series(np.nan, index=pet_series.index)

    for month in range(1, 13):
        month_data_all = pet_accum[pet_accum.index.month == month]

        # Reference period data
        month_data_ref = month_data_all[
            (month_data_all.index.year >= ref_start) &
            (month_data_all.index.year <= ref_end)
        ]

        if len(month_data_ref) < 15:
            continue

        try:
            xi = month_data_all.min() - 0.001

            data_fit = month_data_ref.values - xi
            data_calc = month_data_all.values - xi

            shape, loc, fit_scale = stats.fisk.fit(data_fit, floc=0)

            probabilities = stats.fisk.cdf(data_calc, c=shape, loc=loc, scale=fit_scale)

            probabilities = 1 - probabilities

            probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)

            eddi_values = stats.norm.ppf(probabilities)

            eddi_results.loc[month_data_all.index] = eddi_values

        except Exception as e:
            continue

    return eddi_results.values


def _eddi_worker_function(pet_data: np.ndarray, parameters: Dict) -> np.ndarray:
    """EDDI worker function."""
    return eddi_loglogistic(
        pet_data,
        scale=parameters['scale'],
        start_year=parameters['data_start_year'],
        ref_start=parameters['ref_start'],
        ref_end=parameters['ref_end'],
        periodicity=parameters['periodicity']
    )


# --- Parallel Processing Framework ---

def init_worker(shared_arrays_dict: Dict):
    global _global_shared_arrays
    _global_shared_arrays = shared_arrays_dict


def _apply_along_axis_single(params: Dict[str, Any]):
    """Parallel processing function."""
    func1d = params["func1d"]
    start_index = params["sub_array_start"]
    end_index = params["sub_array_end"]

    shape = _global_shared_arrays[params["output_var_name"]][_KEY_SHAPE]

    pet_array = np.frombuffer(
        _global_shared_arrays[params["var_name_pet"]][_KEY_ARRAY].get_obj()
    ).reshape(shape)

    sub_array_pet = pet_array[start_index:end_index]

    output_array_buffer = _global_shared_arrays[params["output_var_name"]][_KEY_ARRAY]
    computed_array_slice = np.frombuffer(output_array_buffer.get_obj()).reshape(shape)[start_index:end_index]

    for i in range(sub_array_pet.shape[0]):
        for j in range(sub_array_pet.shape[1]):
            pet_timeseries = sub_array_pet[i, j, :]

            if np.all(np.isnan(pet_timeseries)):
                computed_array_slice[i, j, :] = np.nan
                continue

            result_timeseries = func1d(pet_timeseries, parameters=params["args"])
            computed_array_slice[i, j, :] = result_timeseries


def parallel_process(arrays_dict: Dict, input_var_name: str, output_var_name: str, args: Dict):
    """Parallel processing."""
    shape = arrays_dict[output_var_name][_KEY_SHAPE]
    num_processes = min(shape[0], _NUMBER_OF_WORKER_PROCESSES)
    if num_processes == 0:
        return

    indices = np.array_split(range(shape[0]), num_processes)

    chunk_params = []
    for i in range(num_processes):
        params = {
            "func1d": _eddi_worker_function,
            "var_name_pet": input_var_name,
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
        pool.map(_apply_along_axis_single, chunk_params)


def compute_and_write_eddi(params: Dict):
    """Compute and save EDDI."""
    scale = params['scale']
    model_name = params.get('model_name', 'Unknown')

    try:
        ds_pet = xr.open_dataset(params['netcdf_pet'])
    except FileNotFoundError as e:
        _logger.error(f"[{model_name}] File error: {e}")
        return

    try:
        ds_pet = ds_pet.transpose('lat', 'lon', 'time')
    except ValueError:
        _logger.error(f"[{model_name}] Dimension transposition failed.")
        return

    pet_values = ds_pet[params['var_name_pet']].values
    output_shape = pet_values.shape

    try:
        time_val = ds_pet['time'].values[0]
        if hasattr(time_val, 'year'):
            data_start_year = time_val.year
        else:
            data_start_year = pd.to_datetime(time_val).year
        _logger.info(f"[{model_name}] Data start year: {data_start_year}")
    except Exception as e:
        _logger.error(f"[{model_name}] Time parsing failed, defaulting to 1900: {e}")
        data_start_year = 1900

    if params['ref_start'] < data_start_year:
        _logger.warning(
            f"[{model_name}] Warning: The specified reference period start year ({params['ref_start']}) "
            f"is earlier than the data start year ({data_start_year}), which may lead to computation errors."
        )

    _global_shared_arrays.clear()

    _global_shared_arrays[params['var_name_pet']] = {
        _KEY_ARRAY: multiprocessing.Array("d", pet_values.flatten()),
        _KEY_SHAPE: output_shape,
    }

    internal_var_name = "eddi"
    _global_shared_arrays[internal_var_name] = {
        _KEY_ARRAY: multiprocessing.Array("d", int(np.prod(output_shape))),
        _KEY_SHAPE: output_shape,
    }

    calc_args = {
        'scale': scale,
        'data_start_year': data_start_year,
        'periodicity': 'monthly',
        'ref_start': params['ref_start'],
        'ref_end': params['ref_end']
    }

    _logger.info(
        f"[{model_name}] Starting parallel computation of EDDI-{scale} "
        f"(reference period: {params['ref_start']}-{params['ref_end']})..."
    )
    parallel_process(_global_shared_arrays, params['var_name_pet'], internal_var_name, calc_args)

    result_array = np.frombuffer(
        _global_shared_arrays[internal_var_name][_KEY_ARRAY].get_obj()
    ).reshape(output_shape)

    result_array_f32 = result_array.astype(np.float32)

    output_ds = xr.Dataset(
        {internal_var_name: (ds_pet[params['var_name_pet']].dims, result_array_f32)},
        coords={'lat': ds_pet['lat'], 'lon': ds_pet['lon'], 'time': ds_pet['time']}
    )

    output_ds[internal_var_name].attrs['long_name'] = f'EDDI-{scale} ({model_name})'
    output_ds[internal_var_name].attrs['units'] = '1'
    output_ds[internal_var_name].attrs['reference_period'] = f"{params['ref_start']}-{params['ref_end']}"
    output_ds[internal_var_name].attrs['distribution'] = 'Log-Logistic (Fisk)'
    output_ds[internal_var_name].attrs['comment'] = 'Negative values indicate drought (high evaporative demand)'

    output_file_path = os.path.join(params['output_dir'], params['output_filename'])

    encoding = {
        internal_var_name: {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4,
            '_FillValue': np.float32(np.nan),
        },
        'lat': {'dtype': 'float32'},
        'lon': {'dtype': 'float32'},
    }

    output_ds.to_netcdf(output_file_path, encoding=encoding)

    file_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
    _logger.info(f"[{model_name}] Done! Results saved to: {output_file_path} (size: {file_size_mb:.1f} MB)")

    ds_pet.close()


# --- Batch Processing Main Logic ---

def batch_process_models():
    pr_dir = "/home/jijiachen/data_jjc/pet/ssp245"
    base_output_dir = "/home/jijiachen/data_jjc"
    scenario = "ssp245"

    VAR_PET = "pet"  # PET variable name

    SCALES = [1, 3, 6, 9, 12]

    REF_START = 1961
    REF_END = 2024

    PRIORITY_MODEL = "ACCESS-CM2"

    _logger.info("Scanning PET file directory...")

    pet_files = [f for f in os.listdir(pr_dir) if f.endswith(".nc")]
    pet_files.sort(key=lambda x: 0 if PRIORITY_MODEL in x else 1)

    total_files = len(pet_files)
    _logger.info(f"Found {total_files} PET files.")
    _logger.info(f"Scales to compute: {SCALES}")

    for scale in SCALES:
        _logger.info(f"\n{'#'*60}")
        _logger.info(f"### Starting processing for SCALE = {scale}")
        _logger.info(f"{'#'*60}\n")

        scale_folder_name = f"{scale}-month scale EDDI"
        output_dir = os.path.join(base_output_dir, scale_folder_name, scenario)
        os.makedirs(output_dir, exist_ok=True)
        _logger.info(f"Output directory: {output_dir}")

        count = 0
        for pet_file in pet_files:
            try:
                # Filename format: et0_ACCESS-CM2_ssp245_2015-2100_remapped_bc.nc
                parts = pet_file.split('_')
                if len(parts) < 2:
                    continue

                model_name = parts[1]
                count += 1

                pet_path = os.path.join(pr_dir, pet_file)

                output_filename = f"{model_name}_eddi{scale}.nc"
                output_file_path = os.path.join(output_dir, output_filename)

                if os.path.exists(output_file_path):
                    _logger.info(f"[SCALE={scale}] [{count}/{total_files}] {model_name} already exists, skipping.")
                    continue

                _logger.info(
                    f"\n{'='*10} [SCALE={scale}] Processing model [{count}/{total_files}]: {model_name} {'='*10}"
                )

                params = {
                    "netcdf_pet": pet_path,
                    "var_name_pet": VAR_PET,
                    "scale": scale,
                    "output_dir": output_dir,
                    "output_filename": output_filename,
                    "model_name": model_name,
                    "ref_start": REF_START,
                    "ref_end": REF_END
                }

                compute_and_write_eddi(params)

            except Exception as e:
                _logger.error(f"[SCALE={scale}] Unknown error occurred while processing file {pet_file}: {e}")
                import traceback
                traceback.print_exc()

        _logger.info(f"\n[SCALE={scale}] All models processed!")

    _logger.info("\n" + "="*60)
    _logger.info("All scales computation completed!")
    _logger.info("="*60)


if __name__ == "__main__":
    batch_process_models()