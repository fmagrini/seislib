#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inter-Station Dispersion Curves
=============================== 

The below class provides automated routines to calculate
Rayleigh and Love inter-station dispersion curves based on
the continuous seismograms recorded at pairs of receivers.

"""

import os
import itertools as it
from joblib import Parallel, delayed
import numpy as np
from scipy.interpolate import interp1d
from obspy import read
from obspy.geodetics.base import gps2dist_azimuth
from obspy.io.sac.util import SacIOError
import matplotlib
import matplotlib.pyplot as plt
import portalocker
from seislib.utils import rotate_stream, load_pickle, save_pickle, remove_file
from seislib.an import noisecorr, velocity_filter, extract_dispcurve
from seislib.plotting import plot_stations
from seislib.tomography import SeismicTomography
import warnings


def _read(*args, verbose=True, **kwargs):
    try:
        return read(*args, **kwargs)
    except SacIOError:
        if verbose:
            print("SacIOError", *args)


class AmbientNoiseVelocity:
    """
    Class to obtain surface-wave phase velocities out of continuous seismograms,
    via cross-correlation of seismic ambient noise.

    Parameters
    ----------
    src : str
        Absolute path to the directory containing the files associated with
        the continuous seismograms

    savedir : str, optional
        Absolute path to the directory where the results are saved. If not
        provided, its value is set to the parent directory of `src` (i.e.,
        $src/..). All the results will be saved in the directory
        $savedir/an_velocity/$component (see the `component` parameter)

    component : {'Z', 'R', 'T'}
        Either 'Z', 'R', (corresponding to vertically and radially  polarized
        Rayleigh waves, respectively), or 'T' (for Love waves)

    verbose : bool
        Whether or not information on progress is printed in the console


    Attributes
    ----------
    src : str
        Absolute path to the directory containing the files associated with the
        continuous seismograms

    savedir : str
        Absolute path to the directory where the results are saved

    component : {'Z', 'R', 'T'}
        Either 'Z', 'R', or 'T'

    files : list
        List of files to be processed to calculate the dispersion curves

    verbose : bool
        Whether or not information on progress is printed in the console

    stations : dict
        Station coordinates that can be employed to retrieve dispersion
        curves (can be accessed after calling the method :meth:`prepare_data`)


    Examples
    --------
    The following will calculate the dispersion curves for a set of continuous
    seismograms stored in the directory src='/path/to/data'. The data have been
    downloaded using :class:seislib.an.an_downloader.ANDownloader. We extract
    dispersion curves for vertically-polarized Rayleigh waves (on the vertical
    - Z - component).

    >>> from seislib.an import AmbientNoiseVelocity
    >>> an = AmbientNoiseVelocity(
    ...         src=src,
    ...         component='Z')
    >>> print(an)
    RAYLEIGH-WAVE VELOCITY FROM SEISMIC AMBIENT NOISE
    ==================================================
    RECEIVERS: X
    STATION PAIRS AVAILABLE: Y
    DISPERSION CURVES RETRIEVED: 0
    ==================================================
    SOURCE DIR: /path/to/data
    SAVE DIR: /path/to/an_velocity/Z

    In the above printed information X and Y are integers and depend on your data.

    Before extracting the dispersion curves, we need to run::

        an.prepare_data()

    which will extract information on each seismic receiver from the header of
    the sac files. These include (i) station coordinates and (ii) the time window
    spanned by the associated seismogram. The information is saved into two
    separate files, i.e., /path/to/an_velocity/Z/stations.pickle and
    /path/to/an_velocity/Z/timespans.pickle. These include (i) stations
    coordinates and (ii) the time window spanned by the associated seismograms.

    .. note::
        If the files containing the continuous seismograms do not have a sac
        header, or if they have a different extension than .sac, the user can create
        the two above files on his own, before calling the `prepare_data` method.
        (However, each file name should be in the format net.sta.loc.cha', where
        net, sta, and loc (optional) are network, station, and location code,
        respectively, and cha is channel (e.g., BHZ, BHN, or BHE). (If the component
        is R or T, the NE channels will be loaded and rotated during the calculation
        of the phase velocities.) For example, IV.AQU.00.BHZ.

        The stations.pickle file should contain a dictionary object where each key
        corresponds to a station code ($net.$sta) and each value should be a `tuple`
        containing latitude and longitude of the station. For example::

            { net1.sta1 : (lat1, lon1), net1.sta2 : (lat2, lon2)}

        The timespans.pickle file should contain a dictionary object where each key
        corresponds to a station code ($net.$sta) and each value should be a `tuple`
        containing the starttime and endtime (`obspy.UTCDateTime.timestamp`) of the
        continuous seismogram associated with the station. For example::

            {net1.sta1 : (starttime1, endtime1), net1.sta2 : (starttime2, endtime2)}

    After the data have been "prepared", the receivers available can be
    displayed by typing::

        an.plot_stations()

    Now we can calculate the dispersion curves automatically, provided that we
    pass a reference curve (i.e., `ndarray of shape (n, 2)`, where the 1st column
    is frequency, the 2nd is phase velocity). For example::

        an.extract_dispcurves(refcurve,
                              freqmin=0.01,
                              freqmax=0.5,
                              cmin=1.5,
                              cmax=4.5,
                              distmax=2000,
                              window_length=3600,
                              overlap=0.5,
                              min_no_days=30)

    The above will calculate the dispersion curves for all combinations of
    station pairs for which the inter-station distance is < 2000 km and at least
    30 days of simultaneous recordings are available. Cross-correlations will
    be computed in the frequency range 0.01-0.5 Hz on 50%-overlapping time
    windows. The results (`ndarrays of shape (m, 2)`, where the 1st column is
    frequency and the 2nd is phase velocity) will be saved to
    /path/to/an_velocity/Z/dispcurves.

    .. note::
       The function `extract_dispcurves` now accepts the argument `n_cpus`, enabling
       parallel calculation of dispersion curves. Set `n_cpus=1` to run in single-threaded
       mode, or specify the desired number of CPUs to leverage multiprocessing.
    """

    def __init__(self, src, savedir=None, component="Z", verbose=True):
        self.src = src
        savedir = os.path.dirname(src) if savedir is None else savedir
        self.verbose = verbose
        if component not in ["Z", "R", "T"]:
            exception = "@component@ should be either `Z` (Rayleigh, vertical), \
                         `T` (Love, transverse), or `R` (Rayleigh, radial)"
            raise Exception(exception)
        self.component = component
        self.savedir = os.path.join(savedir, "an_velocity", component)
        os.makedirs(self.savedir, exist_ok=True)
        self.files = self.get_files()

    def __repr__(self):
        return str(self)

    def __str__(self):
        phase = "love" if self.component == "T" else "rayleigh"
        if phase == "rayleigh" and self.component == "R":
            phase = phase + " (radial)"
        string = "\n%s-WAVE VELOCITY FROM SEISMIC AMBIENT NOISE" % (phase.upper())
        separators = len(string)
        string += "\n%s" % ("=" * separators)
        stations = set([".".join(i.split(".")[:2]) for i in self.files])
        string += "\nRECEIVERS: %s" % (len(stations))
        try:
            done = len(os.listdir(os.path.join(self.savedir, "dispcurves")))
        except FileNotFoundError:
            done = 0
        pairs = len(stations) * (len(stations) - 1) // 2
        string += "\nSTATION PAIRS AVAILABLE: %s" % (pairs)
        string += "\nDISPERSION CURVES RETRIEVED: %s" % (done)
        string += "\n%s" % ("=" * separators)
        string += "\nSOURCE DIR: %s" % self.src
        string += "\nSAVE DIR: %s" % self.savedir
        return string

    def get_files(self):
        """
        Retrieves the files to be processed for extracting the phase velocities

        Returns
        -------
        files : list of str
            e.g. ['net1.sta1.00.BHZ.sac', 'net1.sta2.00.BHZ.sac']
        """
        files = []
        components = ["Z"] if self.component == "Z" else ["E", "N"]
        for file in sorted(os.listdir(self.src)):
            if file.startswith("."):  # Avoid hidden files
                continue
            channel = file.split(".")[-2][-1]
            if channel in components:
                files.append(file)
        return files

    def get_times_and_coords(self, files):
        """
        Retrieves the geographic coordinates and the starting and ending time
        associated with each continuous seismogram

        Parameters
        ----------
        files: list of str
            Names of the files corresponding with the continuous seismograms,
            located in the `src` directory

        Returns
        -------
        times : dict
            each key corresponds to a station code ($network_code.$station_code)
            and each value is a tuple containing the starttime and endtime
            (in timestamp format, see obspy documentation on UTCDateTime.timestamp)
            of the continuous seismogram associated with the station. For
            example::

                { net1.sta1 : (starttime1, endtime1),
                  net1.sta2 : (starttime2, endtime2),
                  net2.sta3 : (starttime3, endtime3)
                  }

        coords : dict
            each key corresponds to a station code ($network_code.$station_code)
            and each value is a tuple containing latitude and longitude of the
            station. For example::

                { net1.sta1 : (lat1, lon1),
                  net1.sta2 : (lat2, lon2),
                  net2.sta3 : (lat3, lon3)
                  }
        """
        times = {}
        coords = {}
        for file in files:
            station_code = ".".join(file.split(".")[:2])
            if station_code in coords:
                continue
            tr = _read(
                os.path.join(self.src, file), headonly=True, verbose=self.verbose
            )[0]
            lat, lon = tr.stats.sac.stla, tr.stats.sac.stlo
            coords[station_code] = (lat, lon)
            starttime = tr.stats.starttime.timestamp
            endtime = tr.stats.endtime.timestamp
            times[station_code] = (starttime, endtime)
            if self.verbose:
                print(station_code, tr.stats.starttime, tr.stats.endtime)

        return times, coords

    def prepare_data(self, recompute=False):
        """
        Saves to disk the geographic coordinates and the starting and ending
        time associated with each continuous seismogram. These are saved to
        $self.savedir/stations.pickle and $self.savedir/timespans.pickle,
        respectively.

        The stations.pickle file contains a dictionary object where each key
        corresponds to a station code ($network_code.$station_code) and each
        value is a tuple containing latitude and longitude of the station.
        For example::

            { net1.sta1 : (lat1, lon1),
              net1.sta2 : (lat2, lon2),
              net2.sta3 : (lat3, lon3)
              }

        The timespans.pickle file contains a dictionary object similar to the
        above, where each value is a tuple containing the starttime and endtime
        (`obspy.UTCDateTime.timestamp`) of the continuous seismogram associated
        with the station. For example::

            { net1.sta1 : (starttime1, endtime1),
              net1.sta2 : (starttime2, endtime2),
              net2.sta3 : (starttime3, endtime3)
              }

        Parameters
        ----------
        recompute : bool
            If `True`, the station coordinates and times will be removed from
            disk and recalculated. Otherwise (default), if they are present,
            they will be loaded into memory, avoiding any computation. This
            parameter should be set to `True` whenever one has added new files
            to the source directory (and has not updated manually stations.pickle
            and timespans.pickle).

        Notes
        -----
        If `component` is 'Z', only continuous seismograms recorded on the
        vertical component are considered (e.g., BHZ).

        If `component` is either 'R' or 'T', however, for a given station to
        be considered in the calculation of phase velocities both the
        horizontal components (N and E, e.g. BHN and BHE) should be
        available. These will be rotated so as to analyse the transverse
        (T, for Love) and radial (R, for radially-polarized Rayleigh waves)
        components
        """
        savecoords = os.path.join(self.savedir, "stations.pickle")
        savetimes = os.path.join(self.savedir, "timespans.pickle")
        if recompute:
            remove_file(savecoords)
            remove_file(savetimes)
        if not os.path.exists(savecoords) or not os.path.exists(savetimes):
            times, coords = self.get_times_and_coords(self.files)
            save_pickle(savecoords, coords)
            save_pickle(savetimes, times)
        else:
            coords = load_pickle(savecoords)
            times = load_pickle(savetimes)
        self.stations = coords
        self.times = times

    @classmethod
    def convert_to_kms(cls, dispcurve):
        """Converts from m/s to km/s the dispersion curve (if necessary).

        Parameters
        ----------
        dispcurve : ndarray of shape (n, 2)
            The 1st column (typically frequency or period) is ignored. The
            2nd should be velocity. If the 1st value of velocity
            divided by 10 is greater than 1, the 2nd column is divided by
            1000. Otherwise, the dispersion curve is left unchanged.

        Returns
        -------
        dispcurve : ndarray of shape (n, 2)
        """
        if dispcurve[0, 1] / 10 > 1:
            dispcurve = np.column_stack((dispcurve[:, 0], dispcurve[:, 1] / 1000))
        return dispcurve

    def dist_km_azimuth_backazimuth(self, sta1, sta2):
        """Calculates inter-station distance (km), azimuth, and backazimuth associated
        with a pair of receivers

        Parameters
        ----------
        sta1, sta2 : str
            Station codes representing the station pair of interest. These codes must
            correspond to keys in the `stations` attribute of this instance.

        Returns
        -------
        (dist, az, baz) : tuple
        """
        stla1, stlo1 = self.stations[sta1]
        stla2, stlo2 = self.stations[sta2]
        dist, az, baz = gps2dist_azimuth(stla1, stlo1, stla2, stlo2)
        dist /= 1000.0
        return dist, az, baz

    def _plot_dispcurve(
        self,
        dispcurve,
        refcurve,
        crossings,
        xcorr,
        sta1,
        sta2,
        dist,
        days_overlap,
        cmin=1.5,
        cmax=4.5,
        freqmin=0.01,
        freqmax=0.5,
        savefig=None,
        show=True,
    ):
        suptitle = "%s - %s | " % (sta1, sta2)
        suptitle += "Dist: %.1f km | Overlapping days: %.1f" % (dist, days_overlap)
        plt.figure(figsize=plt.figaspect(0.4))
        plt.subplot(1, 2, 1)
        plt.plot(
            crossings[:, 0],
            crossings[:, 1],
            lw=0,
            marker="o",
            label="Zero crossings",
            markersize=5,
        )
        plt.plot(refcurve[:, 0], refcurve[:, 1], "k", lw=1, label="Reference")
        plt.plot(dispcurve[:, 0], dispcurve[:, 1], "r", lw=1.5, label="Retrieved")
        plt.ylim(cmin, cmax)
        plt.ylabel("Phase velocity [km/s]")
        plt.xlim(freqmin, freqmax)
        plt.xlabel("Frequency [Hz]")
        plt.legend(loc="upper right", framealpha=0.9, handlelength=1)
        plt.grid(alpha=0.5)
        plt.title("Dispersion Curve")

        plt.subplot(1, 2, 2)
        plt.plot(xcorr[:, 0], xcorr[:, 1].real, "k", lw=1.5, label="Real")
        plt.plot(xcorr[:, 0], xcorr[:, 1].imag, "gray", lw=1.5, label="Imag")
        plt.xlim(freqmin, freqmax)
        plt.xlabel("Frequency [Hz]")
        plt.grid(alpha=0.5)
        plt.legend(loc="upper right", framealpha=0.9, handlelength=1)
        plt.title("Cross Spectrum")

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.suptitle(suptitle, y=0.98)
        if savefig is not None:
            plt.savefig(savefig, dpi=(200))
        if not show:
            plt.close()
        else:
            plt.show()

    def _read_stream(self, folder, file, component):
        st = _read(os.path.join(folder, file), verbose=self.verbose)
        if st is None:
            return
        if component == "R" or component == "T":
            cha = file[-5]
            new_cha = "N" if cha == "E" else "E"
            new_file = file.replace("%s.sac" % cha, "%s.sac" % new_cha)
            try:
                st += _read(os.path.join(folder, new_file), verbose=self.verbose)
            except TypeError:
                return
        return st

    def _get_trace(self, st, azimuth, component):
        if component == "Z":
            return st[0]
        st = rotate_stream(st, method="NE->RT", back_azimuth=(azimuth + 180) % 360)
        return st.select(component=component)[0]

    def _station_pairs_generator(self, stations, reverse_iteration=False):
        sort = lambda iterable, reverse: sorted(iterable, reverse=reverse)
        station_pairs = (
            pair
            for pair in sort(it.combinations(sorted(stations), 2), reverse_iteration)
        )
        for pair in station_pairs:
            yield pair

    def _load_done(self, file):
        if os.path.exists(file):
            with open(file, "r") as f:
                # Acquire a shared lock for reading
                portalocker.lock(f, portalocker.LOCK_SH)
                done_set = set([i.strip() for i in f])
                # Release the lock
                portalocker.unlock(f)
            return done_set
        else:
            return set()

    def _update_done(self, sta1, sta2, done_file):
        with open(done_file, "a") as f:
            # Acquire an exclusive lock for writing
            portalocker.lock(f, portalocker.LOCK_EX)
            f.write("%s__%s\n" % (sta1, sta2))
            # Release the lock
            portalocker.unlock(f)

    def extract_dispcurves(
        self,
        refcurve,
        freqmin=0.01,
        freqmax=0.5,
        cmin=1.5,
        cmax=4.5,
        distmax=None,
        window_length=3600,
        overlap=0.5,
        min_no_days=30,
        reverse_iteration=False,
        save_xcorr=False,
        plotting=False,
        manual_picking=False,
        show=True,
        n_cpus=1,
    ):
        """
        Automatic extraction of the dispersion curves for all available pairs
        of receivers.

        The results are saved to $self.savedir/dispcurves in .npy format,
        and consist of `ndarrays of shape (n, 2)`, where the 1st column is
        frequency and the 2nd phase velocity (in m/s).

        The routine iterates over all the available combinations of station
        pairs and, for each one, (i) computes the cross spectrum (in the
        frequency domain) by ensemble averaging the cross correlations calculated
        over relatively small (and possibly overlapping, see `overlap`) time
        windows (see `window_length`), (ii) filters the cross-spectrum using
        a "velocity" filter [1]_, and (iii) extracts a smooth dispersion
        curve by comparison of the zero-crossings of the cross-spectrum with
        those of the Bessel function associated with the station pair in
        question [2]_.

        Parameters
        ----------
        refcurve : ndarray of shape (n, 2)
            Reference curve used to extract the dispersion curves. The 1st
            column should be frequency, the 2nd velocity (in either
            km/s or m/s). The reference curve is automatically converted to
            km/s, the physical units employed in the subsequent analysis

        freqmin, freqmax : float
            Minimum and maximum frequency analysed by the algorithm (default
            are 0.01 and 0.5 Hz). The resulting dispersion curves will be limited
            to this frequency range

        cmin, cmax : float
            Estimated velocity range (in km/s) spanned by the dispersion curves
            (default values are 1.5 and 4.5)

        distmax : float
            Maximum inter-station distance (in km), beyond which a given station
            pair is not considered in the calculation of the phase velocities

        window_length : float
            Length of the time windows (in s) used to perform the cross
            correlations

        overlap : float
            Should be >=0 and <1 (strictly smaller than 1). Rules the extent
            of overlap between one time window and the following [3]_. Default
            is 0.5

        min_no_days : int, float
            Minimum number of simultaneous recordings available for a given
            station pair to be considered for the extraction of phase velocity.
            Default is 30

        reverse_iteration : bool
            If `True`, the list of combinations of station pairs will be
            iterated over reversely. This can be useful to run two processes in
            parallel so as to halve the computation times (one function call
            setting `reverse_iteration`=`False`, the other, in a different process,
            setting it to `True`). Default is `False`

        save_xcorr : bool
            If `True`, cross-correlations will be saved in $self.savedir/xcorr

        plotting : bool
            If `True`, a figure is created for each retrieved dispersion curve.
            This is automatically displayed and saved in $self.savedir/figures

        manual_picking : bool
            If `True`, the user is required to pick each dispersion curve manually.
            The picking is carried out through an interactive plot

        show : bool
            If `False`, the figure is closed before being shown. This is only
            relevant when the user has the interactive plot enabled: in that
            case, the figure will not be displayed. Default is `True`

        n_cpus : int
            Number of CPUs to use for parallel processing. When `n_cpus=1`,
            the function runs in single-threaded mode without using joblib.
            When `n_cpus > 1`, joblib is used to parallelize over station pairs.

        Returns
        -------
        None
            The dispersion curves are saved to $self.savedir/dispcurves

        References
        ----------
        .. [1] Magrini & Boschi, (2021), Surface‐Wave Attenuation From Seismic Ambient Noise:
            Numerical Validation and Application, JGR

        .. [2] Kästle et al. 2016, Two-receiver measurements of phase velocity: cross-
            validation of ambient-noise and earthquake-based observations, GJI

        .. [3] Seats et al. 2012, Improved ambient noise correlation functions using
            Welch's method, GJI
        """
        save_pv = os.path.join(self.savedir, "dispcurves")
        save_tmp = os.path.join(self.savedir, "tmp")
        save_done = os.path.join(save_tmp, "DONE.txt")
        save_corr = os.path.join(self.savedir, "xcorr")
        save_fig = os.path.join(self.savedir, "figures")
        os.makedirs(save_pv, exist_ok=True)
        os.makedirs(save_tmp, exist_ok=True)
        if save_xcorr:
            os.makedirs(save_corr, exist_ok=True)
        if plotting:
            os.makedirs(save_fig, exist_ok=True)

        if n_cpus > 1:
            if manual_picking:
                warnings.warn(
                    "Manual picking is not allowed when using `n_cpus`>1 and will be disabled."
                )
                manual_picking = False
            show = False  # Disable showing plots while executing in parallel
            matplotlib.use("Agg")
        # Convert reference curve to km/s
        refcurve = self.convert_to_kms(refcurve)

        # Prepare data
        files_dict = {".".join(i.split(".")[:2]): i for i in self.files}
        horizontal_polarization = False if self.component == "Z" else True
        npairs = int(len(self.stations) * (len(self.stations) - 1) / 2)
        station_pairs = list(
            self._station_pairs_generator(self.stations, reverse_iteration)
        )

        # Load already processed station pairs
        done = self._load_done(save_done)
        station_pairs_to_process = [
            (sta1, sta2)
            for sta1, sta2 in station_pairs
            if "%s__%s" % (sta1, sta2) not in done
        ]

        if self.verbose:
            print("Total station pairs to process:", len(station_pairs_to_process))

        # Common arguments for processing
        common_args = {
            "refcurve": refcurve,
            "files_dict": files_dict,
            "horizontal_polarization": horizontal_polarization,
            "save_pv": save_pv,
            "save_corr": save_corr,
            "save_fig": save_fig,
            "freqmin": freqmin,
            "freqmax": freqmax,
            "cmin": cmin,
            "cmax": cmax,
            "distmax": distmax,
            "window_length": window_length,
            "overlap": overlap,
            "min_no_days": min_no_days,
            "save_xcorr": save_xcorr,
            "plotting": plotting,
            "manual_picking": manual_picking,
            "show": show,
            "save_done": save_done,
        }

        if n_cpus == 1:
            # Single-threaded processing
            for ndone, (sta1, sta2) in enumerate(station_pairs_to_process):
                # Update progress every 100 pairs
                if not ndone % 100:
                    done = self._load_done(save_done)
                    if self.verbose:
                        print("\nPERCENTAGE DONE: %.2f\n" % (len(done) / npairs * 100))
                successful = self._process_station_pair(sta1, sta2, **common_args)
                if successful and self.verbose:
                    print(sta1, sta2)
        else:
            # Multi-processing with joblib
            verbose = 5 if self.verbose else 0
            Parallel(n_jobs=n_cpus, verbose=verbose)(
                delayed(self._process_station_pair)(sta1, sta2, **common_args)
                for sta1, sta2 in station_pairs_to_process
            )

    def _process_station_pair(
        self,
        sta1,
        sta2,
        refcurve,
        files_dict,
        horizontal_polarization,
        save_pv,
        save_corr,
        save_fig,
        freqmin,
        freqmax,
        cmin,
        cmax,
        distmax,
        window_length,
        overlap,
        min_no_days,
        save_xcorr,
        plotting,
        manual_picking,
        show,
        save_done,
    ):
        """
        Process a single station pair to extract the dispersion curve.
        """
        # Check if the station pair has already been processed
        pair_id = "%s__%s" % (sta1, sta2)
        if pair_id in self._load_done(save_done):
            return False

        # Get the SAC files for the station pair
        sac1, sac2 = files_dict.get(sta1), files_dict.get(sta2)
        if sac1 is None or sac2 is None:
            self._update_done(sta1, sta2, save_done)
            return False

        # Get the recording times
        (start1, end1), (start2, end2) = self.times[sta1], self.times[sta2]
        days_overlap = (min([end1, end2]) - max([start1, start2])) / 86400
        if days_overlap < min_no_days:
            self._update_done(sta1, sta2, save_done)
            return False

        # Compute distance and azimuth
        dist, az, baz = self.dist_km_azimuth_backazimuth(sta1, sta2)
        if distmax is not None and dist >= distmax:
            self._update_done(sta1, sta2, save_done)
            return False

        # Read the streams
        st1 = self._read_stream(folder=self.src, file=sac1, component=self.component)
        if st1 is None:
            self._update_done(sta1, sta2, save_done)
            return False
        st2 = self._read_stream(folder=self.src, file=sac2, component=self.component)
        if st2 is None:
            self._update_done(sta1, sta2, save_done)
            return False

        # Get the traces
        tr1 = self._get_trace(st1, azimuth=az, component=self.component)
        tr2 = self._get_trace(st2, azimuth=az, component=self.component)
        dispcurve_file = os.path.join(save_pv, f"{tr1.id}__{tr2.id}.npy")
        if os.path.exists(dispcurve_file):
            self._update_done(sta1, sta2, save_done)
            return False

        xcorr_file = os.path.join(save_corr, f"{tr1.id}__{tr2.id}.npy")
        fig_file = os.path.join(save_fig, f"{tr1.id}__{tr2.id}.png")

        try:
            freq, xcorr = noisecorr(
                tr1, tr2, window_length=window_length, overlap=overlap
            )
            xcorr_smooth = velocity_filter(freq, xcorr, dist, cmin=cmin, cmax=cmax)
            crossings, phase_vel = extract_dispcurve(
                freq,
                xcorr_smooth,
                dist,
                refcurve,
                freqmin=freqmin,
                freqmax=freqmax,
                cmin=cmin,
                cmax=cmax,
                horizontal_polarization=horizontal_polarization,
                manual_picking=manual_picking,
                sta1=sta1,
                sta2=sta2,
            )
            if save_xcorr:
                np.save(xcorr_file, np.column_stack((freq, xcorr)))
        except Exception:
            self._update_done(sta1, sta2, save_done)
            return False

        if plotting:
            self._plot_dispcurve(
                dispcurve=phase_vel,
                refcurve=refcurve,
                crossings=crossings,
                xcorr=np.column_stack((freq, xcorr_smooth)),
                sta1=tr1.id,
                sta2=tr2.id,
                dist=dist,
                days_overlap=days_overlap,
                cmin=cmin,
                cmax=cmax,
                freqmin=freqmin,
                freqmax=freqmax,
                savefig=fig_file,
                show=show,
            )

        phase_vel[:, 1] *= 1000  # Convert velocities to m/s
        np.save(dispcurve_file, phase_vel)
        self._update_done(sta1, sta2, save_done)
        return True

    # def extract_dispcurves(
    #     self,
    #     refcurve,
    #     freqmin=0.01,
    #     freqmax=0.5,
    #     cmin=1.5,
    #     cmax=4.5,
    #     distmax=None,
    #     window_length=3600,
    #     overlap=0.5,
    #     min_no_days=30,
    #     reverse_iteration=False,
    #     save_xcorr=False,
    #     plotting=False,
    #     manual_picking=False,
    #     show=True,
    # ):
    #     """
    #     Automatic extraction of the dispersion curves for all available pairs
    #     of receivers.

    #     The results are saved to $self.savedir/dispcurves in .npy format,
    #     and consist of `ndarrays of shape (n, 2)`, where the 1st column is
    #     frequency and the 2nd phase velocity (in m/s).

    #     The routine iterates over all the available combinations of station
    #     pairs and, for each one, (i) computes the cross spectrum (in the
    #     frequency domain) by ensamble averaging the cross correlations calculated
    #     over relatively small (and possibly overlapping, see `overlap`) time
    #     windows (see `window_length`), (ii) filters the cross-spectrum using
    #     a "velocity" filter [1]_, and (iii) extracts a smooth dispersion
    #     curve by comparison of the zero-crossings of the cross-spectrum with
    #     those of the Bessel function associated with the station pair in
    #     question [2]_.

    #     Parameters
    #     ----------
    #     refcurve : ndarray of shape (n, 2)
    #         Reference curve used to extract the dispersion curves. The 1st
    #         column should be frequency, the 2nd velocity (in either
    #         km/s or m/s). The reference curve is automatically converted to
    #         km/s, the physical units employed in the subsequent analysis

    #     freqmin, freqmax : float
    #         Minimum and maximum frequency analysed by the algorithm (default
    #         are 0.01 and 0.5 Hz). The resulting dispersion curves will be limited
    #         to this frequency range

    #     cmin, cmax : float
    #         Estimated velocity range (in km/s) spanned by the dispersion curves
    #         (default values are 1.5 and 4.5)

    #     distmax : float
    #         Maximum inter-station distance (in km), beyond which a given station
    #         pair is not considered in the calculation of the phase velocities

    #     window_lenght : float
    #         Length of the time windows (in s) used to perform the cross
    #         correlations

    #     overlap : float
    #         Should be >=0 and <1 (strictly smaller than 1). Rules the extent
    #         of overlap between one time window and the following [3]_. Default
    #         is 0.5

    #     min_no_days : int, float
    #         Minimum number of simultaneous recordings available for a given
    #         station pair to be considered for the extraction of phase velocity.
    #         Default is 30

    #     reverse_iteration : bool
    #         If `True`, the list of combinations of station pairs will be
    #         iterated over reversely. This can be useful to run two processes in
    #         parallel so as to halve the computation times (one function call
    #         setting `reverse_iteration`=`False`, the other, in a different process,
    #         setting it to `True`). Default is `False`

    #     save_xcorr : bool
    #         If `True`, cross-correlations will be saved in $self.savedir/xcorr

    #     plotting : bool
    #         If `True`, a figure is created for each retrieved dispersion curve.
    #         This is automatically displayed and saved in $self.savedir/figures

    #     manual_picking : bool
    #         If `True`, the user is required to pick each dispersion curve manually.
    #         The picking is carried out through an interactive plot

    #     show : bool
    #         If `False`, the figure is closed before being shown. This is only
    #         relevant when the user has the interactive plot enabled: in that
    #         case, the figure will not be displayed. Default is `True`

    #     Returns
    #     -------
    #     None
    #         The dispersion curves are saved to $self.savedir/dispcurves

    #     References
    #     ----------
    #     .. [1] Magrini & Boschi, (2021), Surface‐Wave Attenuation From Seismic Ambient Noise:
    #         Numerical Validation and Application, JGR

    #     .. [2] Kästle et al. 2016, Two-receiver measurements of phase velocity: cross-
    #         validation of ambient-noise and earthquake-based observations, GJI

    #     .. [3] Seats et al. 2012, Improved ambient noise correlation functions using
    #         Welch's method, GJI
    #     """
    #     save_pv = os.path.join(self.savedir, "dispcurves")
    #     save_tmp = os.path.join(self.savedir, "tmp")
    #     save_done = os.path.join(save_tmp, "DONE.txt")
    #     save_corr = os.path.join(self.savedir, "xcorr")
    #     save_fig = os.path.join(self.savedir, "figures")
    #     os.makedirs(save_pv, exist_ok=True)
    #     os.makedirs(save_tmp, exist_ok=True)
    #     if save_xcorr:
    #         os.makedirs(save_corr, exist_ok=True)
    #     if plotting:
    #         os.makedirs(save_fig, exist_ok=True)
    #     refcurve = self.convert_to_kms(refcurve)

    #     files_dict = {".".join(i.split(".")[:2]): i for i in self.files}
    #     horizontal_polarization = False if self.component == "Z" else True
    #     npairs = int(len(self.stations) * (len(self.stations) - 1) / 2)
    #     station_pairs = self._station_pairs_generator(self.stations, reverse_iteration)

    #     sta1_code = None
    #     done = self._load_done(save_done)
    #     for ndone, (sta1, sta2) in enumerate(station_pairs):
    #         if "%s__%s" % (sta1, sta2) in done:
    #             continue
    #         if not ndone % 100:
    #             done = self._load_done(save_done)
    #             if self.verbose:
    #                 print("\nPERCENTAGE DONE: %.2f\n" % (len(done) / npairs * 100))

    #         sac1, sac2 = files_dict[sta1], files_dict[sta2]
    #         (start1, end1), (start2, end2) = self.times[sta1], self.times[sta2]
    #         days_overlap = (min([end1, end2]) - max([start1, start2])) / 86400
    #         if days_overlap < min_no_days:
    #             self._update_done(sta1, sta2, save_done)
    #             continue

    #         dist, az, baz = self.dist_km_azimuth_backazimuth(sta1, sta2)
    #         if distmax is not None and dist >= distmax:
    #             self._update_done(sta1, sta2, save_done)
    #             continue

    #         if self.verbose:
    #             print(sta1, sta2)

    #         if sta1_code is None or sta1_code != sta1:
    #             st1 = self._read_stream(
    #                 folder=self.src, file=sac1, component=self.component
    #             )
    #             if st1 is None:
    #                 self._update_done(sta1, sta2, save_done)
    #                 continue
    #             sta1_code = sta1
    #         st2 = self._read_stream(
    #             folder=self.src, file=sac2, component=self.component
    #         )
    #         if st2 is None:
    #             self._update_done(sta1, sta2, save_done)
    #             continue
    #         tr1 = self.get_trace(st1, azimuth=az, component=self.component)
    #         tr2 = self.get_trace(st2, azimuth=az, component=self.component)
    #         dispcurve_file = os.path.join(save_pv, "%s__%s.npy" % (tr1.id, tr2.id))
    #         if os.path.exists(dispcurve_file):
    #             self._update_done(sta1, sta2, save_done)
    #             continue

    #         xcorr_file = os.path.join(save_corr, "%s__%s.npy" % (tr1.id, tr2.id))
    #         fig_file = os.path.join(save_fig, "%s__%s.png" % (tr1.id, tr2.id))

    #         try:
    #             freq, xcorr = noisecorr(
    #                 tr1, tr2, window_length=window_length, overlap=overlap
    #             )
    #             xcorr_smooth = velocity_filter(freq, xcorr, dist, cmin=cmin, cmax=cmax)
    #             crossings, phase_vel = extract_dispcurve(
    #                 freq,
    #                 xcorr_smooth,
    #                 dist,
    #                 refcurve,
    #                 freqmin=freqmin,
    #                 freqmax=freqmax,
    #                 cmin=cmin,
    #                 cmax=cmax,
    #                 horizontal_polarization=horizontal_polarization,
    #                 manual_picking=manual_picking,
    #                 sta1=sta1,
    #                 sta2=sta2,
    #             )
    #             if save_xcorr:
    #                 np.save(xcorr_file, np.column_stack((freq, xcorr)))
    #         except:
    #             self._update_done(sta1, sta2, save_done)
    #             continue

    #         if plotting:
    #             self._plot_dispcurve(
    #                 dispcurve=phase_vel,
    #                 refcurve=refcurve,
    #                 crossings=crossings,
    #                 xcorr=np.column_stack((freq, xcorr_smooth)),
    #                 sta1=tr1.id,
    #                 sta2=tr2.id,
    #                 dist=dist,
    #                 days_overlap=days_overlap,
    #                 cmin=cmin,
    #                 cmax=cmax,
    #                 freqmin=freqmin,
    #                 freqmax=freqmax,
    #                 savefig=fig_file,
    #                 show=show,
    #             )

    #         phase_vel[:, 1] *= 1000  # converts to m/s
    #         np.save(dispcurve_file, phase_vel)
    #         self._update_done(sta1, sta2, save_done)

    def prepare_input_tomography(
        self, savedir, period, min_no_wavelengths=2, outfile="input_%.2fs.txt"
    ):
        r"""
        Prepares a .txt file for each specified period, to be used for
        calculating phase-velocity maps using
        :class:`seislib.tomography.tomography.SeismicTomography`.

        Parameters
        ----------
        savedir : str
            Absolute path to the directory where the file(s) is (are) saved.
            If `savedir` does not exist, it will be created

        period : float, iterable of floats
            Period (or periods) at which the dispersion curves will be
            interpolated (see :meth:`interpolate_dispcurves`)

        min_no_wavelengths : float
            Ratio between the estimated wavelength :math:`\lambda = T c_{ref}`
            of the surface-wave at a given period *T* and interstation distance
            :math:`\Delta`. Whenever this ratio is > `min_no_wavelength`,
            the period in question is not used to retrieve a dispersion measurement.
            Values < 1 are suggested against. Default is 2

        outfile : str
            Format for the file names. It must include either %s or %.Xf (where
            X is integer), since this will be replaced by each period analysed
            (one for file)

        Examples
        --------
        Assume you calculated dispersion curves from your data, which are
        located in /path/to/data, and you had initialized your
        class:`AmbientNoiseVelocity` instance, to calculate inter-station
        dispersion curves, as::

            an = AmbientNoiseVelocity(src=SRC,
                                      component='Z')
            an.prepare_data()

        You can, even if the computation of the dispersion curves is not
        finished, use the above instance to extract the information needed
        to produce phase-velocity maps as follows::

            savedir = /arbitrary/path/to/savedir
            periods = [5, 6, 8, 10, 12, 15, 20]
            an.prepare_input_tomography(savedir=savedir,
                                        period=periods)

        The above will save one txt file for each specified period.
        """
        os.makedirs(savedir, exist_ok=True)
        period = np.array([period]) if np.isscalar(period) else np.array(period)
        codes, coords, velocity = self.interpolate_dispcurves(1 / period)
        dist = SeismicTomography.gc_distance(*coords.T)
        wavelength = velocity * period
        ratios = dist.reshape(-1, 1) / wavelength
        velocity_final = np.where(ratios > min_no_wavelengths, velocity, np.nan)

        for iperiod, p in enumerate(period):
            save = os.path.join(savedir, outfile % p)
            vel = velocity_final[:, iperiod]
            notnan = np.flatnonzero(~np.isnan(vel))
            if self.verbose:
                print("Measurements at %.2fs:" % p, notnan.size)
            if notnan.size:
                np.savetxt(save, np.column_stack((coords[notnan], vel[notnan])))

    def interpolate_dispcurves(self, frequency):
        r"""
        Interpolates the dispersion curves found at $self.savedir/dispcurves
        at the specified frequency. (No extrapolation is made.)

        Parameters
        ----------
        frequency : float, iterable of floats
            Frequency (or frequencies) at which the dispersion curves will be
            interpolated

        Returns
        -------
        codes : ndarray of shape (n, 2)
            Codes associated with the station pairs for which a dispersion curve
            has been calculated

        coords : ndarray of shape (n, 4)
            Coordinates (lat1, lon1, lat2, lon2) of the station pairs corresponding
            to the station codes

        measurements : ndarray of shape (n, f)
            Phase velocities calculated for each station pair in `coords` at
            the specified `frequency`. f is the number of frequencies

            .. note::
                *measurements* could contain nans

        Examples
        --------
        Assume you calculated dispersion curves from your data, which are
        located in /path/to/data, and you had initialized your
        class:`AmbientNoiseVelocity` instance, to calculate inter-station
        dispersion curves, as::

            an = AmbientNoiseVelocity(src=SRC,
                                      component='Z')
            an.prepare_data()

        Even if the computation of the dispersion curves is not yet
        finished, you can use the above instance to interpolate all the
        available dispersion curves as follows::

            frequency = [1/30, 1/20, 1/10, 1/5]
            codes, coords, measurements = an.interpolate_dispcurves(frequency)
        """
        src = os.path.join(self.savedir, "dispcurves")
        files = [i for i in sorted(os.listdir(src)) if i.endswith("npy")]
        files_size = len(files)
        freq_size = 1 if np.isscalar(frequency) else len(frequency)
        measurements = np.zeros((files_size, freq_size))
        coords = np.zeros((files_size, 4))
        codes = []
        for i, file in enumerate(files):
            if self.verbose and not i % int(0.05 * files_size + 1):
                print("FILES PROCESSED: %d/%d" % (i, files_size))
            freq, vel = np.load(os.path.join(src, file)).T
            interp_vel = interp1d(freq, vel, bounds_error=False)(frequency)
            measurements[i] = interp_vel
            code1, code2 = file.split(".npy")[0].split("__")
            sta1 = ".".join(code1.split(".")[:2])
            sta2 = ".".join(code2.split(".")[:2])
            coords[i] = (*self.stations[sta1], *self.stations[sta2])
            codes.append((sta1, sta2))
        return np.array(codes), coords, measurements

    def plot_stations(
        self,
        ax=None,
        show=True,
        oceans_color="water",
        lands_color="land",
        edgecolor="k",
        projection="Mercator",
        resolution="110m",
        color_by_network=True,
        legend_dict={},
        **kwargs,
    ):
        r"""Maps the seismic receivers for which data are available

        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not `None`, the receivers are plotted on the `GeoAxesSubplot` instance.
            Otherwise, a new figure and `GeoAxesSubplot` instance is created

        show : bool
            If `True`, the plot is shown. Otherwise, a `GeoAxesSubplot` instance is
            returned. Default is `True`

        oceans_color, lands_color : str
            Color of oceans and lands. The arguments are ignored if ax is not
            `None`. Otherwise, they are passed to `cartopy.feature.NaturalEarthFeature`
            (to the argument 'facecolor'). Defaults are 'water' and 'land'

        edgecolor : str
            Color of the boundaries between, e.g., lakes and land. The argument
            is ignored if ax is not `None`. Otherwise, it is passed to
            `cartopy.feature.NaturalEarthFeature` (to the argument 'edgecolor').
            Default is 'k' (black)

        projection : str
            Name of the geographic projection used to create the `GeoAxesSubplot`.
            (Visit the `cartopy website
            <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_
            for a list of valid projection names.) If ax is not None, `projection`
            is ignored. Default is 'Mercator'

        resolution : {'10m', '50m', '110m'}
            Resolution of the Earth features displayed in the figure. Passed to
            `cartopy.feature.NaturalEarthFeature`. Valid arguments are '110m',
            '50m', '10m'. Default is '110m'. Ignored if `ax` is not `None`.

        color_by_network : bool
            If `True`, each seismic network will have a different color in the
            resulting map, and a legend will be displayed. Otherwise, all
            stations will have the same color. Default is `True`

        legend_dict : dict, optional
            Dictionary of keyword arguments passed to `matplotlib.pyplot.legend`

        **kwargs : dict, optional
            Additional keyword arguments passed to `matplotlib.pyplot.scatter`

        Returns
        -------
        If `show` is True, None, else `ax`, i.e. the GeoAxesSubplot
        """
        return plot_stations(
            stations=self.stations,
            ax=ax,
            show=show,
            oceans_color=oceans_color,
            lands_color=lands_color,
            edgecolor=edgecolor,
            projection=projection,
            resolution=resolution,
            color_by_network=color_by_network,
            legend_dict=legend_dict,
            **kwargs,
        )
