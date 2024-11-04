from vedo import settings as vsettings
import sys

DEBUG = False  # set to True to see more detailed logs

# ------------------------------- vedo settings ------------------------------ #

vsettings.pointSmoothing = True
vsettings.lineSmoothing = False
vsettings.polygonSmoothing = False
vsettings.immediateRendering = False

vsettings.useDepthPeeling = True
vsettings.alphaBitPlanes = 1
vsettings.maxNumberOfPeels = 12
vsettings.occlusionRatio = 0.2
vsettings.multiSamples = 0 if sys.platform == "darwin" else 8

vsettings.useSSAO = True

# For transparent background with screenshots
vsettings.screenshotTransparentBackground = True  # vedo for transparent bg
vsettings.useFXAA = False  # This needs to be false for transparent bg


# --------------------------- brainrender settings --------------------------- #

BACKGROUND_COLOR = "white"
DEFAULT_ATLAS = "allen_mouse_25um"  # default atlas
DEFAULT_CAMERA = "three_quarters"  # Default camera settings (orientation etc. see brainrender.camera.py)
INTERACTIVE = False  # rendering interactive ?
LW = 2  # e.g. for silhouettes
ROOT_COLOR = [0.8, 0.8, 0.8]  # color of the overall brain model's actor
ROOT_ALPHA = 0.2  # transparency of the overall brain model's actor'
SCREENSHOT_SCALE = 1
SHADER_STYLE = "shiny"  # affects the look of rendered brain regions: [cartoon, metallic, plastic, shiny, glossy]
SHOW_AXES = False
WHOLE_SCREEN = True  # If true render window is full screen
OFFSCREEN = False
