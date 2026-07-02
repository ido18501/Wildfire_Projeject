"""
Updated data loader for Sim2Real-Fire dataset.
Matches actual folder structure discovered in simulation_data.

Structure:
    data_root/
    └── 0014_00426/                      # Scene folder
        ├── Topography_Map/
        │   ├── Elevation.tif
        │   ├── Slope.tif
        │   └── Aspect.tif
        ├── Fuel_Map/
        │   ├── FBFM13.tif
        │   ├── Canopy_Cover.tif
        │   └── ...
        ├── Vegetation_Map/
        │   └── ...
        ├── Satellite_Images_Mask/
        │   ├── 0014_00001/              # Fire event
        │   │   ├── out1.jpg ... out72.jpg
        │   └── ...
        └── Weather_Data/
            ├── 0014_00001.txt
            └── ...
"""

import os
import glob
import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class Sim2RealFireLoader:
    """Load Sim2Real-Fire dataset with actual folder structure."""
    
    # Which TIF files to use from each category
    TOPO_FILES = ['Elevation.tif', 'Slope.tif', 'Aspect.tif']
    
    # Primary fuel files (most relevant for fire spread)
    FUEL_FILES = ['FBFM13.tif', 'Canopy_Cover.tif', 'Canopy_Height.tif', 
                  'Canopy_Bulk_Density.tif']
    
    # Primary vegetation files
    VEG_FILES = ['Existing_Vegetation_Type.tif', 'Existing_Vegetation_Cover.tif']
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: Path containing scene folders (e.g., .../simulation_data/data/)
        """
        self.data_root = data_root
        self.scenes = self._discover_scenes()
        
    def _discover_scenes(self) -> dict:
        """Find all scenes and their fire events. Handles flat and nested scene folders."""
        scenes = {}

        for entry in os.listdir(self.data_root):
            outer_path = os.path.join(self.data_root, entry)
            if not os.path.isdir(outer_path):
                continue

            candidate_paths = [outer_path]
            for child in os.listdir(outer_path):
                child_path = os.path.join(outer_path, child)
                if os.path.isdir(child_path):
                    candidate_paths.append(child_path)

            scene_path = None
            for cand in candidate_paths:
                mask_dir = os.path.join(cand, "Satellite_Images_Mask")
                weather_dir = os.path.join(cand, "Weather_Data")
                if os.path.isdir(mask_dir) and os.path.isdir(weather_dir):
                    scene_path = cand
                    break

            if scene_path is None:
                continue

            mask_dir = os.path.join(scene_path, "Satellite_Images_Mask")
            weather_dir = os.path.join(scene_path, "Weather_Data")

            events = []
            for event_folder in os.listdir(mask_dir):
                event_path = os.path.join(mask_dir, event_folder)
                if os.path.isdir(event_path):
                    weather_file = os.path.join(weather_dir, f"{event_folder}.txt")
                    if os.path.exists(weather_file):
                        events.append(event_folder)

            if events:
                scenes[entry] = {
                    "path": scene_path,
                    "events": sorted(events),
                }

        return scenes

    def list_scenarios(self, scene_filter: list = None) -> list:
        """Return list of (scene_id, event_id) tuples.
        
        Args:
            scene_filter: Optional list of scene IDs to include.
                         If None, returns all scenes.
        """
        scenarios = []
        for scene_id, info in self.scenes.items():
            if scene_filter is not None and scene_id not in scene_filter:
                continue
            for event_id in info['events']:
                scenarios.append((scene_id, event_id))
        return scenarios
    
    def load_scenario(self, scene_id: str, event_id: str) -> dict:
        """
        Load a complete fire scenario.
        
        Returns:
            dict with keys:
                'topography': dict of numpy arrays (elevation, slope, aspect)
                'fuel': dict of numpy arrays
                'vegetation': dict of numpy arrays
                'weather': numpy array [T, num_features]
                'masks': numpy array [T, H, W] binary masks
                'timestamps': list of hours from start
                'scene_id': str
                'event_id': str
        """
        scene_path = self.scenes[scene_id]['path']
        
        # Load static covariates
        topo = self._load_tif_folder(
            os.path.join(scene_path, 'Topography_Map'), 
            self.TOPO_FILES
        )
        fuel = self._load_tif_folder(
            os.path.join(scene_path, 'Fuel_Map'),
            self.FUEL_FILES
        )
        veg = self._load_tif_folder(
            os.path.join(scene_path, 'Vegetation_Map'),
            self.VEG_FILES
        )
        
        # Load fire event data
        mask_dir = os.path.join(scene_path, 'Satellite_Images_Mask', event_id)
        weather_file = os.path.join(scene_path, 'Weather_Data', f"{event_id}.txt")
        
        masks, timestamps = self._load_mask_sequence(mask_dir)
        weather = self._load_weather(weather_file)
        
        return {
            'topography': topo,
            'fuel': fuel,
            'vegetation': veg,
            'weather': weather,
            'masks': masks,
            'timestamps': timestamps,
            'scene_id': scene_id,
            'event_id': event_id
        }
    
    def _load_tif_folder(self, folder: str, file_list: list = None) -> dict:
        """Load all TIF files from a folder.

        file_list is kept for backward compatibility but ignored.
        """
        data = {}
        if not os.path.isdir(folder):
            return data

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(".tif"):
                continue

            filepath = os.path.join(folder, filename)
            key = filename.replace(".tif", "").lower()
            data[key] = self._load_tif(filepath)

        return data
    
    def _load_tif(self, filepath: str) -> np.ndarray:
        """Load a GeoTIFF file."""
        if HAS_RASTERIO:
            with rasterio.open(filepath) as src:
                return src.read(1).astype(np.float32)
        else:
            # Fallback to PIL (works for simple TIFs)
            img = Image.open(filepath)
            return np.array(img, dtype=np.float32)
    
    def _load_mask_sequence(self, mask_dir: str) -> tuple:
        """
        Load sequence of fire masks.
        
        Returns:
            masks: [T, H, W] boolean array
            timestamps: list of hours from start
        """
        # Find all out*.jpg files
        mask_files = glob.glob(os.path.join(mask_dir, 'out*.jpg'))
        
        # Sort by numeric index
        def get_index(f):
            basename = os.path.basename(f)
            return int(basename.replace('out', '').replace('.jpg', ''))
        
        mask_files = sorted(mask_files, key=get_index)
        
        masks = []
        for f in mask_files:
            img = Image.open(f).convert('L')  # Grayscale
            arr = np.array(img)
            # Convert to binary (burned = True)
            # Assuming white/bright = burned
            binary = arr > 127
            masks.append(binary)
        
        masks = np.stack(masks, axis=0)
        timestamps = list(range(len(masks)))  # Hourly: 0, 1, 2, ... hours
        
        return masks, timestamps
    
    def _load_weather(self, filepath: str) -> np.ndarray:
        """
        Load weather data file.
        
        Columns: Year Month Day HHMM Temp Humidity Precip WindSpeed WindDir Extra
        
        Returns:
            weather: [T, 6] array with (temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos, precip)
        """
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    temp = float(parts[4])
                    humidity = float(parts[5])
                    precip = float(parts[6])
                    wind_speed = float(parts[7])
                    wind_dir_deg = float(parts[8])
                    
                    # Convert wind direction to sin/cos
                    wind_dir_rad = np.deg2rad(wind_dir_deg)
                    wind_dir_sin = np.sin(wind_dir_rad)
                    wind_dir_cos = np.cos(wind_dir_rad)
                    
                    data.append([temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos, precip])
        
        return np.array(data, dtype=np.float32)


def extract_arrival_times(masks: np.ndarray, timestamps: list) -> np.ndarray:
    """
    Convert mask sequence to arrival time field.
    
    Args:
        masks: [T, H, W] boolean array (True = burned)
        timestamps: [T] list of times in hours
    
    Returns:
        arrival_times: [H, W] array, value = time when pixel first burned
                       inf for never-burned pixels
    """
    T, H, W = masks.shape
    arrival = np.full((H, W), np.inf, dtype=np.float32)
    
    for t_idx, t in enumerate(timestamps):
        # Pixels that are burned at this time but weren't assigned yet
        newly_assigned = masks[t_idx] & np.isinf(arrival)
        arrival[newly_assigned] = float(t)
    
    return arrival


def find_ignition_point(masks: np.ndarray) -> tuple:
    """Find the ignition point from the first non-empty mask."""
    for t in range(len(masks)):
        if masks[t].any():
            # Centroid of first burned region
            ys, xs = np.where(masks[t])
            return int(np.mean(ys)), int(np.mean(xs))
    return None


class CovariateAdapter:
    """Convert Sim2Real covariates to model input format.

    feature_group:
        baseline = original 6 spatial / 5 global features
        full     = expanded available Sim2Real features
    """

    def __init__(self, feature_group: str = "baseline"):
        self.feature_group = feature_group

        full_spatial = [
            "elevation",
            "slope",
            "aspect_sin",
            "aspect_cos",
            "fbfm13",
            "canopy_cover",
            "canopy_height",
            "canopy_bulk_density",
            "existing_vegetation_type",
            "existing_vegetation_cover",
        ]

        all_available_spatial = [
            "elevation",
            "slope",
            "aspect_sin",
            "aspect_cos",
            "fbfm13",
            "fbfm40",
            "canopy_cover",
            "canopy_height",
            "canopy_bulk_density",
            "canopy_base_height",
            "fuel_disturbance",
            "fuel_vegetation_cover",
            "fuel_vegetation_height",
            "fuel_vegetation_type",
            "existing_vegetation_type",
            "existing_vegetation_cover",
            "existing_vegetation_height",
            "succession_classes",
            "vegetation_condition_class",
            "vegetation_departure",
        ]

        full_global = [
            "temperature",
            "humidity",
            "wind_speed",
            "wind_dir_sin",
            "wind_dir_cos",
            "precipitation",
        ]

        topo_spatial = ["elevation", "slope", "aspect_sin", "aspect_cos"]
        topo_no_elev = ["slope", "aspect_sin", "aspect_cos"]
        topo_no_slope_aspect = ["elevation"]

        wind_global = ["wind_speed", "wind_dir_sin", "wind_dir_cos"]
        dryness_global = ["temperature", "humidity", "precipitation"]
        weather_global = dryness_global + wind_global

        fuel_type_spatial = ["fbfm13"]
        canopy_spatial = ["canopy_cover", "canopy_height", "canopy_bulk_density"]
        veg_spatial = ["existing_vegetation_type", "existing_vegetation_cover"]
        fuel_veg_spatial = fuel_type_spatial + canopy_spatial + veg_spatial

        omitted_spatial = [
            "fbfm40",
            "canopy_base_height",
            "fuel_disturbance",
            "fuel_vegetation_cover",
            "fuel_vegetation_height",
            "fuel_vegetation_type",
            "existing_vegetation_height",
            "succession_classes",
            "vegetation_condition_class",
            "vegetation_departure",
        ]

        physics_spatial = topo_spatial + fuel_veg_spatial
        all_spatial = physics_spatial + omitted_spatial

        groups = {
            # references
            "baseline": (
                ["elevation", "slope", "aspect_sin", "aspect_cos", "fbfm13", "canopy_cover"],
                ["temperature", "humidity", "wind_speed", "wind_dir_sin", "wind_dir_cos"],
            ),
            "physics_plus": (physics_spatial, weather_global),
            "full": (physics_spatial, weather_global),
            "all_available": (all_spatial, weather_global),

            # single blocks
            "topography_only": (topo_spatial, ["constant_global"]),
            "weather_only": (["constant_spatial"], weather_global),
            "fuel_veg_only": (fuel_veg_spatial, ["constant_global"]),
            "omitted_only": (omitted_spatial, ["constant_global"]),

            # block pairs
            "topo_weather": (topo_spatial, weather_global),
            "topo_fuel": (topo_spatial + fuel_veg_spatial, ["constant_global"]),
            "topo_omitted": (topo_spatial + omitted_spatial, ["constant_global"]),
            "weather_fuel": (fuel_veg_spatial, weather_global),
            "weather_omitted": (omitted_spatial, weather_global),
            "fuel_omitted": (fuel_veg_spatial + omitted_spatial, ["constant_global"]),

            # block triples
            "topo_weather_fuel": (topo_spatial + fuel_veg_spatial, weather_global),
            "topo_weather_omitted": (topo_spatial + omitted_spatial, weather_global),
            "topo_fuel_omitted": (topo_spatial + fuel_veg_spatial + omitted_spatial, ["constant_global"]),
            "weather_fuel_omitted": (fuel_veg_spatial + omitted_spatial, weather_global),

            # WRF-guided removals from physics_plus
            "no_wind": (physics_spatial, dryness_global),
            "no_dryness": (physics_spatial, wind_global),
            "no_weather": (physics_spatial, ["constant_global"]),
            "no_fuel_type": (topo_spatial + canopy_spatial + veg_spatial, weather_global),
            "no_canopy": (topo_spatial + fuel_type_spatial + veg_spatial, weather_global),
            "no_vegetation": (topo_spatial + fuel_type_spatial + canopy_spatial, weather_global),
            "no_topography": (fuel_veg_spatial, weather_global),
            "no_elevation": (topo_no_elev + fuel_veg_spatial, weather_global),
            "no_slope_aspect": (topo_no_slope_aspect + fuel_veg_spatial, weather_global),
            "no_precipitation": (physics_spatial, ["temperature", "humidity", "wind_speed", "wind_dir_sin", "wind_dir_cos"]),
            "no_humidity": (physics_spatial, ["temperature", "precipitation", "wind_speed", "wind_dir_sin", "wind_dir_cos"]),
            "no_temperature": (physics_spatial, ["humidity", "precipitation", "wind_speed", "wind_dir_sin", "wind_dir_cos"]),

            # add-one omitted features onto physics_plus
            "plus_fbfm40": (physics_spatial + ["fbfm40"], weather_global),
            "plus_canopy_base_height": (physics_spatial + ["canopy_base_height"], weather_global),
            "plus_fuel_disturbance": (physics_spatial + ["fuel_disturbance"], weather_global),
            "plus_fuel_vegetation_cover": (physics_spatial + ["fuel_vegetation_cover"], weather_global),
            "plus_fuel_vegetation_height": (physics_spatial + ["fuel_vegetation_height"], weather_global),
            "plus_fuel_vegetation_type": (physics_spatial + ["fuel_vegetation_type"], weather_global),
            "plus_existing_vegetation_height": (physics_spatial + ["existing_vegetation_height"], weather_global),
            "plus_succession_classes": (physics_spatial + ["succession_classes"], weather_global),
            "plus_vegetation_condition_class": (physics_spatial + ["vegetation_condition_class"], weather_global),
            "plus_vegetation_departure": (physics_spatial + ["vegetation_departure"], weather_global),
        }

        if feature_group not in groups:
            raise ValueError(f"Unknown feature_group={feature_group}. Available: {sorted(groups)}")

        self.spatial_feature_names, self.global_feature_names = groups[feature_group]

        self.spatial_channels = len(self.spatial_feature_names)
        self.global_channels = len(self.global_feature_names)

    def __call__(self, scenario: dict) -> tuple:
        import torch

        elev = scenario["topography"].get("elevation")
        if elev is None:
            raise ValueError("Elevation data required")

        H, W = elev.shape
        spatial_list = []

        for name in self.spatial_feature_names:
            if name == "elevation":
                arr = elev
                spatial_list.append(self._robust_normalize(arr))

            elif name == "slope":
                arr = scenario["topography"].get("slope", np.zeros((H, W), dtype=np.float32))
                spatial_list.append(self._robust_normalize(arr))

            elif name == "aspect_sin":
                aspect = scenario["topography"].get("aspect", np.zeros((H, W), dtype=np.float32))
                spatial_list.append(np.sin(np.deg2rad(aspect)).astype(np.float32))

            elif name == "aspect_cos":
                aspect = scenario["topography"].get("aspect", np.zeros((H, W), dtype=np.float32))
                spatial_list.append(np.cos(np.deg2rad(aspect)).astype(np.float32))

            elif name in scenario.get("fuel", {}):
                spatial_list.append(self._robust_normalize(scenario["fuel"][name]))

            elif name in scenario.get("vegetation", {}):
                spatial_list.append(self._robust_normalize(scenario["vegetation"][name]))

            else:
                spatial_list.append(np.zeros((H, W), dtype=np.float32))

        spatial = np.stack(spatial_list, axis=0).astype(np.float32)

        weather = scenario["weather"]
        global_raw = weather.mean(axis=0)

        global_values = {
            "temperature": (global_raw[0] - 75) / 15,
            "humidity": (global_raw[1] - 50) / 25,
            "wind_speed": (global_raw[2] - 10) / 8,
            "wind_dir_sin": global_raw[3],
            "wind_dir_cos": global_raw[4],
            "precipitation": global_raw[5] if len(global_raw) > 5 else 0.0,
            "constant_global": 0.0,
        }

        global_feat = np.array(
            [global_values[name] for name in self.global_feature_names],
            dtype=np.float32
        )

        global_feat = np.clip(global_feat, -10, 10)
        global_feat = np.nan_to_num(global_feat, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(spatial), torch.tensor(global_feat)

    def _robust_normalize(self, arr: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.std() < 1e-10:
            return np.zeros_like(arr, dtype=np.float32)

        median = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25

        if iqr > 1e-10:
            normalized = (arr - median) / (iqr * 0.7413 + 1e-10)
        else:
            mad = np.median(np.abs(arr - median))
            if mad > 1e-10:
                normalized = (arr - median) / (mad * 1.4826 + 1e-10)
            else:
                normalized = arr - median

        normalized = np.clip(normalized, -clip_range, clip_range)
        return normalized.astype(np.float32)


# =============================================================================
# Quick test script
# =============================================================================

if __name__ == '__main__':
    import sys
    
    # Default path
    data_root = '/path/to/sim2real_data/simulation_data/data'
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    
    print(f"Loading from: {data_root}")
    
    # Initialize loader
    loader = Sim2RealFireLoader(data_root)
    
    # List scenes
    print(f"\nFound {len(loader.scenes)} scenes:")
    for scene_id, info in loader.scenes.items():
        print(f"  {scene_id}: {len(info['events'])} fire events")
    
    # List all scenarios
    scenarios = loader.list_scenarios()
    print(f"\nTotal scenarios: {len(scenarios)}")
    
    if scenarios:
        # Load first scenario
        scene_id, event_id = scenarios[0]
        print(f"\nLoading scenario: {scene_id}/{event_id}")
        
        data = loader.load_scenario(scene_id, event_id)
        
        print(f"\nTopography layers: {list(data['topography'].keys())}")
        for k, v in data['topography'].items():
            print(f"  {k}: {v.shape}, range [{v.min():.1f}, {v.max():.1f}]")
        
        print(f"\nFuel layers: {list(data['fuel'].keys())}")
        for k, v in data['fuel'].items():
            print(f"  {k}: {v.shape}, range [{v.min():.1f}, {v.max():.1f}]")
        
        print(f"\nVegetation layers: {list(data['vegetation'].keys())}")
        
        print(f"\nWeather: {data['weather'].shape}")
        print(f"  Temp range: [{data['weather'][:,0].min():.1f}, {data['weather'][:,0].max():.1f}] °F")
        print(f"  Wind speed range: [{data['weather'][:,2].min():.1f}, {data['weather'][:,2].max():.1f}]")
        
        print(f"\nMasks: {data['masks'].shape}")
        print(f"  Timestamps: {len(data['timestamps'])} hours")
        
        # Extract arrival times
        arrival = extract_arrival_times(data['masks'], data['timestamps'])
        burned = np.isfinite(arrival)
        print(f"\nArrival times: {arrival.shape}")
        print(f"  Burned pixels: {burned.sum()} / {burned.size} ({100*burned.mean():.1f}%)")
        print(f"  Time range: [{arrival[burned].min():.0f}, {arrival[burned].max():.0f}] hours")
        
        # Find ignition
        ignition = find_ignition_point(data['masks'])
        print(f"\nIgnition point: {ignition}")
        
        # Test covariate adapter
        print("\n--- Testing CovariateAdapter ---")
        adapter = CovariateAdapter()
        spatial_feat, global_feat = adapter(data)
        print(f"Spatial features: {spatial_feat.shape}")
        print(f"Global features: {global_feat.shape}")
        print(f"Global values: {global_feat.numpy()}")
        
        print("\n✓ All tests passed!")
