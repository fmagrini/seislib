v1.1.2 (26/11/2024)
--------------------
- Bug fix: `AmbientNoiseVelocity.prepare_data` now avoids processing hidden files such as .DS_Store, which cause errors.


v1.1.1 (22/11/2024)
--------------------
- `AmbientNoiseVelocity.prepare_data` now allows processing of channels without an H as second letter, e.g., DPZ.


v1.1.0 (20/11/2024)
--------------------
- New feature: multiprocessing capabilities enabled in `seislib.an.AmbientNoiseVelocity.extract_dispcurves`
- Added dependencies: `joblib`, `portalocker`

v1.0.0 (19/11/2024)
--------------------
- API change: removed the `seislib.colormaps` module
- Added dependencies: `cmcrameri`
