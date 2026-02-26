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
        """Find all scenes and their fire events."""
        scenes = {}
        
        # Look for scene folders (pattern: XXXX_XXXXX)
        for entry in os.listdir(self.data_root):
            scene_path = os.path.join(self.data_root, entry)
            if not os.path.isdir(scene_path):
                continue
            
            # Check if it has the expected subfolders
            mask_dir = os.path.join(scene_path, 'Satellite_Images_Mask')
            weather_dir = os.path.join(scene_path, 'Weather_Data')
            
            if os.path.isdir(mask_dir) and os.path.isdir(weather_dir):
                # Find all fire events in this scene
                events = []
                for event_folder in os.listdir(mask_dir):
                    event_path = os.path.join(mask_dir, event_folder)
                    if os.path.isdir(event_path):
                        # Check corresponding weather file exists
                        weather_file = os.path.join(weather_dir, f"{event_folder}.txt")
                        if os.path.exists(weather_file):
                            events.append(event_folder)
                
                if events:
                    scenes[entry] = {
                        'path': scene_path,
                        'events': sorted(events)
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
    
    def _load_tif_folder(self, folder: str, file_list: list) -> dict:
        """Load specified TIF files from a folder."""
        data = {}
        for filename in file_list:
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                key = filename.replace('.tif', '').lower()
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
            weather: [T, 5] array with (temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos)
        """
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    temp = float(parts[4])
                    humidity = float(parts[5])
                    wind_speed = float(parts[7])
                    wind_dir_deg = float(parts[8])
                    
                    # Convert wind direction to sin/cos
                    wind_dir_rad = np.deg2rad(wind_dir_deg)
                    wind_dir_sin = np.sin(wind_dir_rad)
                    wind_dir_cos = np.cos(wind_dir_rad)
                    
                    data.append([temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos])
        
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
    
    Uses robust normalization (median/IQR) matching the fire_training pipeline.
    """
    
    def __init__(self, spatial_channels: int = 6, global_channels: int = 5):
        self.spatial_channels = spatial_channels
        self.global_channels = global_channels
    
    def __call__(self, scenario: dict) -> tuple:
        """
        Convert scenario to (spatial_features, global_features).
        
        Returns:
            spatial_feat: [C_spatial, H, W] tensor
            global_feat: [C_global] tensor (time-averaged weather, normalized)
        """
        import torch
        
        # Get grid size from elevation
        elev = scenario['topography'].get('elevation')
        if elev is None:
            raise ValueError("Elevation data required")
        H, W = elev.shape
        
        # Build spatial features with ROBUST normalization
        spatial = np.zeros((self.spatial_channels, H, W), dtype=np.float32)
        
        # Channel 0: Elevation (robust normalized)
        spatial[0] = self._robust_normalize(elev)
        
        # Channel 1: Slope (robust normalized)
        slope = scenario['topography'].get('slope', np.zeros((H, W)))
        spatial[1] = self._robust_normalize(slope)
        
        # Channel 2-3: Aspect as sin/cos (already bounded [-1, 1])
        aspect = scenario['topography'].get('aspect', np.zeros((H, W)))
        aspect_rad = np.deg2rad(aspect)
        spatial[2] = np.sin(aspect_rad)
        spatial[3] = np.cos(aspect_rad)
        
        # Channel 4: Fuel model (robust normalized)
        fuel = scenario['fuel'].get('fbfm13', np.ones((H, W)))
        spatial[4] = self._robust_normalize(fuel)
        
        # Channel 5: Canopy cover (robust normalized)
        canopy = scenario['fuel'].get('canopy_cover', np.zeros((H, W)))
        spatial[5] = self._robust_normalize(canopy)
        
        # Build global features (time-averaged weather, normalized with known scales)
        weather = scenario['weather']  # [T, 5]: temp, humidity, wind_speed, wind_sin, wind_cos
        global_feat = weather.mean(axis=0)  # [5] average over time
        
        # Normalize using domain knowledge (similar to your KNOWN_SCALES approach)
        # [0]: temp (Fahrenheit) - typical range 60-90, use mean=75, std=15
        global_feat[0] = (global_feat[0] - 75) / 15
        # [1]: humidity (0-100) - use mean=50, std=25
        global_feat[1] = (global_feat[1] - 50) / 25
        # [2]: wind_speed (mph) - typical range 0-30, use mean=10, std=8
        global_feat[2] = (global_feat[2] - 10) / 8
        # [3,4]: wind_sin, wind_cos - already in [-1, 1], keep as-is
        # (They're already normalized angular components)
        
        # Clip to prevent extreme values (matching your clip_range=10)
        global_feat = np.clip(global_feat, -10, 10)
        
        # Handle any NaNs
        global_feat = np.nan_to_num(global_feat, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.tensor(spatial), torch.tensor(global_feat)
    
    def _robust_normalize(self, arr: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        """
        Robustly normalize array using median/IQR (matching fire_training pipeline).
        
        This is more robust to outliers than min-max normalization.
        """
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for constant array
        if arr.std() < 1e-10:
            return np.zeros_like(arr)
        
        # Robust normalization using median and IQR
        median = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        
        if iqr > 1e-10:
            # Scale by IQR (0.7413 factor approximates std for normal distribution)
            normalized = (arr - median) / (iqr * 0.7413 + 1e-10)
        else:
            # Fall back to MAD (median absolute deviation)
            mad = np.median(np.abs(arr - median))
            if mad > 1e-10:
                normalized = (arr - median) / (mad * 1.4826 + 1e-10)
            else:
                normalized = arr - median
        
        # Clip to prevent extreme values
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
