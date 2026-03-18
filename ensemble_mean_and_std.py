import numpy as np
import xarray as xr
import glob
import os

# ===== Configuration =====
base_dir = '/home/jijiachen/SPEI'
scales = [1, 3, 6, 9, 12]
scenario = 'ssp126'

# Build all folder paths
data_dirs = [f'{base_dir}/{scale}-month scale SPEI/{scenario}' for scale in scales]

print("="*60)
print("Multi-scale SPEI Ensemble Statistics Processing")
print("="*60)
print(f"Processing scales: {scales}")
print(f"Scenario: {scenario}")
print("="*60)

# ===== Loop through each scale =====
for scale, data_dir in zip(scales, data_dirs):
    print(f"\n{'#'*60}")
    print(f"### Processing SPEI-{scale}")
    print(f"### Directory: {data_dir}")
    print(f"{'#'*60}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: Directory does not exist, skipping - {data_dir}")
        continue
    
    # Find model files
    model_files = sorted(glob.glob(f'{data_dir}/*_spei{scale}.nc'))
    
    if len(model_files) == 0:
        print(f"Warning: No model files found, skipping")
        continue
    
    print(f"Found {len(model_files)} model files")

    # ===== 1. Get dimension information =====
    ds_ref = xr.open_dataset(model_files[0])
    spei_ref = ds_ref['spei']

    print(f"Original dimension order: {spei_ref.dims}")

    # Uniformly transpose to (time, lat, lon)
    spei_ref = spei_ref.transpose('time', 'lat', 'lon')
    print(f"Dimension order after transpose: {spei_ref.dims}")

    time_future = spei_ref.sel(time=slice('2025', '2100'))['time']
    lat = ds_ref['lat']
    lon = ds_ref['lon']

    n_time = len(time_future)
    n_lat = len(lat)
    n_lon = len(lon)
    n_models = len(model_files)

    print(f"Future period: {n_time} time steps, {n_lat} x {n_lon} grid, {n_models} models")
    ds_ref.close()

    # ===== 2. Initialize accumulators =====
    sum_array = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)
    sum_sq_array = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)
    count_array = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)

    # ===== 3. Accumulate model by model =====
    print("\nAccumulating model by model...")
    for i, f in enumerate(model_files):
        ds = xr.open_dataset(f)['spei']
        
        # Transpose to uniform order (time, lat, lon)
        ds = ds.transpose('time', 'lat', 'lon')
        
        # Select future period
        ds_future = ds.sel(time=slice('2025', '2100'))
        data = ds_future.values

        # Create valid value mask
        valid_mask = ~np.isnan(data)
        data_filled = np.where(valid_mask, data, 0.0)
        
        sum_array += data_filled
        sum_sq_array += data_filled ** 2
        count_array += valid_mask.astype(np.float64)
        
        ds.close()
        print(f"  [{i+1}/{n_models}] Completed: {os.path.basename(f)}")

    # ===== 4. Calculate mean and standard deviation =====
    print("\nCalculating statistics...")
    count_array = np.where(count_array == 0, np.nan, count_array)

    mean_array = sum_array / count_array
    variance_array = sum_sq_array / count_array - mean_array ** 2
    variance_array = np.maximum(variance_array, 0)
    std_array = np.sqrt(variance_array)

    # ===== 5. Get observation period data =====
    print("Reading observation period data...")
    ds_obs = xr.open_dataset(model_files[0])['spei']
    ds_obs = ds_obs.transpose('time', 'lat', 'lon')
    ds_obs = ds_obs.sel(time=slice('1961', '2024'))
    time_obs = ds_obs['time']

    print(f"Observation period: {len(time_obs)} time steps")

    # ===== 6. Build and save datasets =====
    print("\nBuilding output dataset...")

    # Future period mean DataArray
    da_future_mean = xr.DataArray(
        mean_array, 
        dims=['time', 'lat', 'lon'],
        coords={'time': time_future, 'lat': lat, 'lon': lon}
    )

    # Concatenate observation + future periods
    spei_full = xr.concat([ds_obs, da_future_mean], dim='time')

    # Save mean
    output_mean = f'{data_dir}/SPEI{scale}_ensemble_mean_1961-2100.nc'
    ds_out = spei_full.to_dataset(name='spei')
    ds_out['spei'].attrs = {
        'long_name': f'SPEI-{scale} (Obs 1961-2024 + Multi-model mean 2025-2100)', 
        'units': '-',
        'scale': scale,
        'scenario': scenario,
        'n_models': n_models
    }
    ds_out.to_netcdf(output_mean)
    print(f"Saved: {output_mean}")

    # Standard deviation
    output_std = f'{data_dir}/SPEI{scale}_ensemble_std_2025-2100.nc'
    da_std = xr.DataArray(
        std_array,
        dims=['time', 'lat', 'lon'], 
        coords={'time': time_future, 'lat': lat, 'lon': lon}
    )
    ds_std = da_std.to_dataset(name='spei_std')
    ds_std['spei_std'].attrs = {
        'long_name': f'Inter-model standard deviation of SPEI-{scale}', 
        'units': '-',
        'scale': scale,
        'scenario': scenario,
        'n_models': n_models
    }
    ds_std.to_netcdf(output_std)
    print(f"Saved: {output_std}")
    
    # Close dataset
    ds_obs.close()
    
    print(f"\n[SPEI-{scale}] Processing completed!")

# ===== Final summary =====
print("\n" + "="*60)
print("All scales processed successfully!")
print("="*60)

print("\nOutput files summary:")
for scale in scales:
    data_dir = f'{base_dir}/{scale}-month scale SPEI/{scenario}'
    if os.path.exists(data_dir):
        print(f"\nSPEI-{scale}:")
        mean_file = f'{data_dir}/SPEI{scale}_ensemble_mean_1961-2100.nc'
        std_file = f'{data_dir}/SPEI{scale}_ensemble_std_2025-2100.nc'
        
        if os.path.exists(mean_file):
            size_mb = os.path.getsize(mean_file) / (1024*1024)
            print(f"  ✓ {os.path.basename(mean_file)} ({size_mb:.1f} MB)")
        
        if os.path.exists(std_file):
            size_mb = os.path.getsize(std_file) / (1024*1024)
            print(f"  ✓ {os.path.basename(std_file)} ({size_mb:.1f} MB)")