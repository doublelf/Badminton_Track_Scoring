"""Check the local sensitivity of the court mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from badminton.court.mapper import CourtMapper
from badminton.court.reference import CourtSpec

spec = CourtSpec()
corners = [(435, 269), (839, 269), (976, 642), (306, 642)]
mapper = CourtMapper(corners, spec.corner_order, spec)

print("corner sanity:")
for corner in corners:
    print("   ", corner, "->", [round(v, 3) for v in mapper.image_to_court(corner)])

print("\nlocal scale around a far-baseline point (636, 271):")
base = mapper.image_to_court((636.0, 271.0))
for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1), (0, 1.0), (0, -1.0), (1.0, 0), (5.0, 0)]:
    p = mapper.image_to_court((636.0 + dx, 271.0 + dy))
    print(
        f"   d=({dx:+5.1f},{dy:+5.1f})px -> court=({p[0]:8.3f},{p[1]:7.3f})  "
        f"delta=({p[0] - base[0]:+7.3f},{p[1] - base[1]:+7.3f}) m"
    )

print("\nlocal scale around the middle of the court (640, 450):")
base = mapper.image_to_court((640.0, 450.0))
for dx, dy in [(1.0, 0), (0, 1.0)]:
    p = mapper.image_to_court((640.0 + dx, 450.0 + dy))
    print(
        f"   d=({dx:+5.1f},{dy:+5.1f})px -> court=({p[0]:8.3f},{p[1]:7.3f})  "
        f"delta=({p[0] - base[0]:+7.3f},{p[1] - base[1]:+7.3f}) m"
    )
