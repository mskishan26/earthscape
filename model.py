"""
Backward-compatibility re-export.

All model classes now live in the models/ package.
Import from models/ directly for new code.
"""

from models.midfusion import MidFusionResNet, adapt_conv1  # noqa: F401
from models.rgb_backbone import RGBBackbone  # noqa: F401
