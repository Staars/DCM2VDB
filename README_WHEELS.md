# Embedding pydicom in the Extension

## How It Works

The extension **embeds** the pydicom wheel file directly in the `extension/wheels/` directory. When the extension loads, it automatically installs pydicom using Blender's embedded Python interpreter via `pip install`. This works in **all environments**, including air-gapped systems, because no internet connection is required.

## Current Setup

✅ **Already configured!** The wheel is bundled in `extension/wheels/pydicom-3.0.1-py3-none-any.whl`

The extension will automatically:
1. Check if pydicom is available
2. If not, install it from the bundled wheel using Blender's Python
3. Import and use pydicom normally

## Updating pydicom (if needed)

To update to a newer version of pydicom:

```bash
# Download the new wheel
python3 -m pip download pydicom==<VERSION> -d extension/wheels --only-binary=:all: --python-version 3.11 --platform any

# Update the wheel path in blender_manifest.toml
```

Or use the provided script:
```bash
./download_wheels.sh
```

## Technical Details

- **Installation method**: Uses `subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", wheel_path])`
- **Blender Python**: Blender 4.5 uses Python 3.11
- **Platform**: pydicom is pure Python, so the wheel works on all platforms (Windows, macOS, Linux)
- **No internet required**: The wheel is bundled, so it works in restricted/air-gapped environments
- **Automatic**: Installation happens transparently when the extension is enabled

## Why This Approach?

Many workspaces (hospitals, research facilities, corporate environments) have:
- No internet access (air-gapped networks)
- Restricted pip/PyPI access
- Firewall restrictions
- Security policies preventing external downloads

By embedding the wheel files, the extension works everywhere without requiring users to manually install dependencies.
