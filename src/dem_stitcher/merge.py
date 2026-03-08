import math
import warnings
from typing import Union

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import copy_first, merge
from rasterio.transform import rowcol
from shapely.geometry import box
from tqdm import tqdm


def _snap_bounds_to_pixel_grid(
    transform: Affine, extent: list[float]
) -> tuple[float, float, float, float]:
    """Snap extent outward to the pixel grid of the given transform.

    `rasterio.merge` truncates to the inner pixel grid, which can lose up to one
    row/column at each edge.  Snapping outward to enclosing pixel boundaries
    avoids this. It does have `target_aligned_pixels`, which nominally tries to
    do the same thing we're doing here, but for non-whole-pixel-algined grids
    like glo_30, it will try to change the extent to integers.
    """
    xmin, ymin, xmax, ymax = extent
    row_min, col_min = rowcol(transform, xmin, ymax, op=math.floor)
    row_max, col_max = rowcol(transform, xmax, ymin, op=math.ceil)
    snap_xmin, snap_ymax = transform * (col_min, row_min)
    snap_xmax, snap_ymin = transform * (col_max, row_max)
    return (snap_xmin, snap_ymin, snap_xmax, snap_ymax)


def merge_tile_datasets_within_extent(
    datasets: Union[list[rasterio.DatasetReader], list[str]],
    extent: list[float],
    resampling: str = 'nearest',
    nodata: float = None,
    n_threads: int = 5,
    dtype: Union[str, np.dtype] = None,
) -> tuple[np.ndarray, dict]:
    # 4269 is North American epsg similar to 4326 and used for 3dep DEM
    inputs_str = isinstance(datasets[0], str)
    if inputs_str:
        datasets_objs = [rasterio.open(ds_path) for ds_path in datasets]
    else:
        datasets_objs = datasets

    try:
        if datasets_objs[0].profile['crs'] not in [CRS.from_epsg(4326), CRS.from_epsg(4269)]:
            raise ValueError('CRS must be epgs:4326')

        extent_box = box(*extent)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            datasets_filtered = [
                ds
                for ds in datasets_objs
                if (
                    box(*ds.bounds).intersects(extent_box)
                    and (box(*ds.bounds).intersection(extent_box).geom_type == 'Polygon')
                )
            ]

        if not datasets_filtered:
            raise ValueError('No datasets intersect requested extent')

        src_profile = datasets_filtered[0].profile.copy()
        dst_dtype = src_profile['dtype'] if dtype is None else dtype
        dst_nodata = src_profile['nodata'] if nodata is None else nodata

        merge_bounds = _snap_bounds_to_pixel_grid(
            datasets_filtered[0].transform, extent
        )

        with tqdm(total=len(datasets_filtered), desc='Reading tile imagery') as pbar:

            def copy_first_with_progress(*args, **kwargs):
                copy_first(*args, **kwargs)
                pbar.update(1)

            arr_merged, merged_transform = merge(
                datasets_filtered,
                bounds=merge_bounds,
                resampling=Resampling[resampling],
                method=copy_first_with_progress,
                nodata=dst_nodata,
                dtype=dst_dtype,
            )

        prof_merged = src_profile.copy()
        prof_merged['transform'] = merged_transform
        prof_merged['count'] = arr_merged.shape[0]
        prof_merged['height'] = arr_merged.shape[1]
        prof_merged['width'] = arr_merged.shape[2]
        prof_merged['nodata'] = dst_nodata
        prof_merged['dtype'] = dst_dtype
        return arr_merged, prof_merged
    finally:
        if inputs_str:
            for ds in datasets_objs:
                ds.close()


def merge_arrays_with_geometadata(
    arrays: list[np.ndarray],
    profiles: list[dict],
    resampling: str = 'bilinear',
    nodata: float = None,
    dtype: str = None,
    method: str = 'first',
) -> tuple[np.ndarray, dict]:
    """Merge arrays in memory with geometadata.

    Parameters
    ----------
    arrays : list[np.ndarray]
        Arrays to merge (must be in the same CRS)
    profiles : list[dict]
        Geometadata for each array
    resampling : str, optional
        See acceptable values rasterio.enums.Resampling, by default 'bilinear'
    nodata : float, optional
        Nodata value to be inserted into merged profile. If None, uses the nodata value from the first profile,
        by default None
    dtype : str, optional
        Dtype to be inserted into merged profile. If None, uses the dtype from the first profile, by default None
    method : str, optional
        See acceptable values in rasterio.merge.merge, by default 'first'

    Returns
    -------
    tuple[np.ndarray, dict]
        Merged array and profile

    Raises
    ------
    ValueError
        * If arrays are not in BIP format
        * If arrays have different number of dimensions (i.e. 2 or 3)
        * If number of profiles is not the same as number of arrays
    """
    n_dim = arrays[0].shape
    if len(n_dim) not in [2, 3]:
        raise ValueError('Currently arrays must be in BIP formati.e. channels x height x width or flat array')
    if len(set([len(arr.shape) for arr in arrays])) != 1:
        raise ValueError('All arrays must have same number of dimensions i.e. 2 or 3')

    if len(n_dim) == 2:
        arrays_input = [arr[np.newaxis, ...] for arr in arrays]
    else:
        arrays_input = arrays

    if (len(arrays)) != (len(profiles)):
        raise ValueError('Length of arrays and profiles needs to be the same')

    memfiles = [MemoryFile() for p in profiles]
    datasets = [mfile.open(**p) for (mfile, p) in zip(memfiles, profiles)]
    [ds.write(arr) for (ds, arr) in zip(datasets, arrays_input)]

    if dtype is None:
        dst_dtype = profiles[0]['dtype']
    else:
        dst_dtype = dtype

    if nodata is None:
        dst_nodata = profiles[0]['nodata']
    else:
        dst_nodata = nodata

    merged_arr, merged_trans = merge(
        datasets, resampling=Resampling[resampling], method=method, nodata=dst_nodata, dtype=dst_dtype
    )

    prof_merged = profiles[0].copy()
    prof_merged['transform'] = merged_trans
    prof_merged['count'] = merged_arr.shape[0]
    prof_merged['height'] = merged_arr.shape[1]
    prof_merged['width'] = merged_arr.shape[2]
    prof_merged['nodata'] = dst_nodata
    prof_merged['dtype'] = dst_dtype

    [ds.close() for ds in datasets]
    [mfile.close() for mfile in memfiles]

    return merged_arr, prof_merged
