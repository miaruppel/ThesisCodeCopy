"""Define important constants used throughout the pipeline."""
import enum

FOVINFLATIONSCALE3D = 1000.0

_DEFAULT_SLICE_THICKNESS = 3.125
_DEFAULT_PIXEL_SIZE = 3.125
_DEFAULT_MAX_IMG_VALUE = 255.0

_RAYLEIGH_FACTOR = 0.66
GRYOMAGNETIC_RATIO = 11.777  # MHz/T
_VEN_PERCENTILE_RESCALE = 99.0
_VEN_PERCENTILE_THRESHOLD_SEG = 80
_PROTON_PERCENTILE_RESCALE = 99.8

KCO_ALPHA = 11.2  # membrane
KCO_BETA = 14.6  # RBC
VA_ALPHA = 1.43


class IOFields(object):
    """General IOFields constants."""

    BANDWIDTH = "bandwidth"
    BIASFIELD_KEY = "biasfield_key"
    DWELL_TIME = "dwell_time"
    FA_DIS = "fa_dis"
    FA_GAS = "fa_gas"
    FIDS = "fids"
    FIDS_DIS = "fids_dis"
    FIDS_GAS = "fids_gas"
    FIELD_STRENGTH = "field_strength"
    FLIP_ANGLE_FACTOR = "flip_angle_factor"
    FOV = "fov"
    FOV = "fov"
    FREQ_CENTER = "freq_center"
    FREQ_EXCITATION = "freq_excitation"
    GIT_BRANCH = "git_branch"
    GRAD_DELAY_X = "grad_delay_x"
    GRAD_DELAY_Y = "grad_delay_y"
    GRAD_DELAY_Z = "grad_delay_z"
    IMAGE = "image"
    INFLATION = "inflation"
    KERNEL_SHARPNESS = "kernel_sharpness"
    MASK_REG_NII = "mask_reg_nii"
    N_FRAMES = "n_frames"
    N_SKIP_END = "n_skip_end"
    N_SKIP_START = "n_skip_start"
    N_DIS_REMOVED = "n_dis_removed"
    N_GAS_REMOVED = "n_gas_removed"
    NPTS = "npts"
    ORIENTATION = "orientation"
    OUTPUT_PATH = "output_path"
    PIXEL_SIZE = "pixel_size"
    PROCESS_DATE = "process_date"
    PROTOCOL_NAME = "protocol_name"
    PROTON_DICOM_DIR = "proton_dicom_dir"
    PROTON_REG_NII = "proton_reg_nii"
    RAMP_TIME = "ramp_time"
    RAW_PROTON_MONTAGE = "raw_proton_montage"
    REGISTRATION_KEY = "registration_key"
    REMOVEOS = "removeos"
    REMOVE_NOISE = "remove_noise"
    SCAN_DATE = "scan_date"
    SCAN_TYPE = "scan_type"
    SEGMENTATION_KEY = "segmentation_key"
    SHAPE_FIDS = "shape_fids"
    SHAPE_IMAGE = "shape_image"
    SITE = "site"
    SLICE_THICKNESS = "slice_thickness"
    SOFTWARE_VERSION = "software_version"
    SUBJECT_ID = "subject_id"
    T2_CORRECTION_FACTOR = "t2_correction_factor"
    TE90 = "te90"
    TR = "tr"
    TR_DIS = "tr_dis"
    TRAJ = "traj"
    TRAJ_DIS = "traj_dis"
    TRAJ_GAS = "traj_gas"
    VEN_COR_MONTAGE = "bias_cor_ven_montage"
    VEN_CV = "ven_cv"
    VEN_DEFECT = "ven_defect"
    VEN_HIGH = "ven_high"
    VEN_HIST = "ven_hist"
    VEN_LOW = "ven_low"
    VEN_MEAN = "ven_mean"
    VEN_MEDIAN = "ven_median"
    VEN_MONTAGE = "ven_montage"
    VEN_SKEW = "ven_skewness"
    VEN_SNR = "ven_snr"
    VEN_STD = "ven_std"
    VENT_DICOM_DIR = "vent_dicom_dir"


class OutputPaths(object):
    """Output file names."""

    GRE_MASK_NII = "GRE_mask.nii"
    GRE_REG_PROTON_NII = "GRE_regproton.nii"
    GRE_VENT_RAW_NII = "GRE_ventraw.nii"
    GRE_VENT_COR_NII = "GRE_ventcor.nii"
    GRE_VENT_BINNING_NII = "GRE_ventbinning.nii"
    VEN_RAW_MONTAGE_PNG = "raw_ven_montage.png"
    PROTON_REG_MONTAGE_PNG = "raw_proton_montage.png"
    VEN_COR_MONTAGE_PNG = "bias_cor_ven_montage.png"
    VEN_COLOR_MONTAGE_PNG = "ven_montage.png"
    VEN_HIST_PNG = "ven_hist.png"
    REPORT_CLINICAL_HTML = "report_clinical.html"
    TEMP_GRE_CLINICAL_HTML = "temp_clinical_gre.html"
    HTML_TMP = "html_tmp"
    REPORT_CLINICAL = "report_clinical"


class CNNPaths(object):
    """Paths to saved model files."""


class ImageType(enum.Enum):
    """Segmentation flags."""

    VENT = "vent"
    UTE = "ute"


class SegmentationKey(enum.Enum):
    """Segmentation flags."""

    CNN_VENT = "cnn_vent"
    CNN_PROTON = "cnn_proton"
    MANUAL_VENT = "manual_vent"
    MANUAL_PROTON = "manual_proton"
    SKIP = "skip"
    THRESHOLD_VENT = "threshold_vent"


class RegistrationKey(enum.Enum):
    """Registration flags.

    Defines how and if registration is performed. Options:
    PROTON2GAS: Register ANTs to register proton image (moving) to gas image (fixed).
        Also uses the transformation and applies on the mask if segmented on proton
        image.
    MASK2GAS: Register ANTs to register mask (moving) to gas image (fixed).
        Also uses the transformation and applies on the proton image.
    MANUAL: Read in Nifti file of manually registered proton image.
    SKIP: Skip registration entirely.
    """

    MANUAL = "manual"
    MASK2GAS = "mask2gas"
    PROTON2GAS = "proton2gas"
    SKIP = "skip"


class BiasfieldKey(enum.Enum):
    """Biasfield correction flags.

    Defines how and if biasfield correction is performed. Options:
    N4ITK: Use N4ITK bias field correction.
    SKIP: Skip bias field ocrrection entirely.
    """

    N4ITK = "n4itk"
    SKIP = "skip"
    RF_DEPOLARIZATION = "rf_depolarization"


class ScanType(enum.Enum):
    """Scan type."""

    NORMALDIXON = "normal"
    MEDIUMDIXON = "medium"
    FASTDIXON = "fast"


class Site(enum.Enum):
    """Site name."""

    DUKE = "duke"
    UVA = "uva"


class Platform(enum.Enum):
    """Scanner platform."""

    SIEMENS = "siemens"


class TrajType(object):
    """Trajectory type."""

    SPIRAL = "spiral"
    HALTON = "halton"
    HALTONSPIRAL = "haltonspiral"
    SPIRALRANDOM = "spiralrandom"
    ARCHIMEDIAN = "archimedian"
    GOLDENMEAN = "goldenmean"


class Orientation(object):
    """Image orientation."""

    CORONAL = "coronal"
    AXIAL = "axial"
    TRANSVERSE = "transverse"
    CORONAL_CCHMC = "coronal_cchmc"


class DCFSpace(object):
    """Defines the DCF space."""

    GRIDSPACE = "gridspace"
    DATASPACE = "dataspace"


class Methods(object):
    """Defines the method to calculate the RBC oscillation image."""

    ELEMENTWISE = "elementwise"
    MEAN = "mean"
    SMOOTH = "smooth"
    BSPLINE = "bspline"


class BinningMethods(object):
    """Define the method to preprocess and bin RBC oscillation image."""

    BANDPASS = "bandpass"
    FIT_SINE = "fitsine"
    NONE = "none"
    THRESHOLD_STRETCH = "threshold_stretch"
    THRESHOLD = "threshold"
    PEAKS = "peaks"


class StatsIOFields(object):
    """Statistic IO Fields."""

    SUBJECT_ID = "subject_id"
    INFLATION = "inflation"
    RBC_M_RATIO = "rbc_m_ratio"
    SCAN_DATE = "scan_date"
    PROCESS_DATE = "process_date"
    SNR_RBC = "snr_rbc"
    SNR_MEMBRANE = "snr_membrane"
    SNR_VENT = "snr_vent"
    PCT_RBC_HIGH = "pct_rbc_high"
    PCT_RBC_LOW = "pct_rbc_low"
    PCT_RBC_DEFECT = "pct_rbc_defect"
    PCT_MEMBRANE_HIGH = "pct_membrane_high"
    PCT_MEMBRANE_LOW = "pct_membrane_low"
    PCT_MEMBRANE_DEFECT = "pct_membrane_defect"
    PCT_VENT_HIGH = "pct_vent_high"
    PCT_VENT_LOW = "pct_vent_low"
    PCT_VENT_DEFECT = "pct_vent_defect"
    MEAN_RBC = "mean_rbc"
    MEAN_MEMBRANE = "mean_membrane"
    MEAN_VENT = "mean_vent"
    MEDIAN_RBC = "median_rbc"
    MEDIAN_MEMBRANE = "median_membrane"
    MEDIAN_VENT = "median_vent"
    STDDEV_RBC = "stddev_rbc"
    STDDEV_MEMBRANE = "stddev_membrane"
    STDDEV_VENT = "stddev_vent"
    N_POINTS = "n_points"
    DLCO = "dlco"
    KCO = "kco"
    ALVEOLAR_VOLUME = "alveolar_volume"


class MatIOFields(object):
    """Mat file IO Fields."""

    SUBJECT_ID = "subject_id"
    IMAGE_RBC_OSC = "image_rbc_osc"


class VENTHISTOGRAMFields(object):
    """Ventilation histogram fields."""

    COLOR = (0.4196, 0.6824, 0.8392)
    XLIM = 1.0
    YLIM = 0.07
    NUMBINS = 50
    REFERENCE_FIT = (0.04074, 0.619, 0.196)


class RBCHISTOGRAMFields(object):
    """Ventilation histogram fields."""

    COLOR = (247.0 / 255, 96.0 / 255, 111.0 / 255)
    XLIM = 1.2
    YLIM = 0.1
    NUMBINS = 50
    REFERENCE_FIT = (0.06106, 0.471, 0.259)


class MEMBRANEHISTOGRAMFields(object):
    """Membrane histogram fields."""

    COLOR = (0.4, 0.7608, 0.6471)
    XLIM = 2.5
    YLIM = 0.18
    NUMBINS = 70
    REFERENCE_FIT = (0.0700, 0.736, 0.278)


class PDFOPTIONS(object):
    """PDF Options dict."""

    VEN_PDF_OPTIONS = {
        "page-width": 256,  # 320,
        "page-height": 160,  # 160,
        "margin-top": 1,
        "margin-right": 0.1,
        "margin-bottom": 0.1,
        "margin-left": 0.1,
        "dpi": 300,
        "encoding": "UTF-8",
        "enable-local-file-access": None,
    }


class NormalizationMethods(object):
    """Image normalization methods."""

    MAX = "max"
    PERCENTILE_MASKED = "percentile_masked"
    PERCENTILE = "percentile"
    MEAN = "mean"


class CMAP(object):
    """Maps of binned values to color values."""

    RBC_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [0, 0.57, 0.71],
        6: [0, 0, 1],
    }

    VENT_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [0, 0.57, 0.71],
        6: [0, 0, 1],
    }

    MEMBRANE_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [184.0 / 255.0, 226.0 / 255.0, 145.0 / 255.0],
        6: [243.0 / 255.0, 205.0 / 255.0, 213.0 / 255.0],
        7: [225.0 / 255.0, 129.0 / 255.0, 162.0 / 255.0],
        8: [197.0 / 255.0, 27.0 / 255.0, 125.0 / 255.0],
    }
