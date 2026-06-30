v1.2.1 (30/6/2026)
------------------
- Bug fix: replaced deprecated NumPy aliases (`np.row_stack` and `np.in1d`) with `np.vstack` and `np.isin` for compatibility with recent NumPy versions.


v1.2.0 (4/5/2026)
------------------
- Enhancement: added optional subcell ray-hit validation to `SeismicTomography.refine_parameterization`.
- Bug fix: tomography coefficient compilation now raises an error when measurement endpoints fall outside the grid.


v1.1.4 (2/12/2025)
------------------
- Enhancement: addedm method `Grid.from_state`


v1.1.3 (2/12/2025)
------------------
- API change: `Grid` is now public
- Enhancement: `EqualAreaGrid` and `RegularGrid` now enable saving and loading grid instances through the methods `save` and `load`

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
