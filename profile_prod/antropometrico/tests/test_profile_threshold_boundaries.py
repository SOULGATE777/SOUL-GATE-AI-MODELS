"""Boundary checks for profile anthropometric labels. Does not load a .pth model."""

import math
import sys
import types
from pathlib import Path

import pytest

# The measurement methods only need numpy. Stub the ML stack so this file
# can run without torch or a checkpoint.
if "torch" not in sys.modules:
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")

    class _Module:
        def __init__(self, *args, **kwargs):
            pass

    nn.Module = _Module
    nn.Sequential = lambda *args, **kwargs: None
    functional = types.ModuleType("torch.nn.functional")
    torch.Tensor = type("Tensor", (), {})
    torch.nn = nn
    torch.nn.functional = functional
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional

    torchvision = types.ModuleType("torchvision")
    torchvision.models = types.SimpleNamespace(resnet50=lambda **kwargs: None)
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.models"] = torchvision.models

    sys.modules["cv2"] = types.ModuleType("cv2")
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.models.profile_anthropometric_pipeline import ProfileAnthropometricPipeline


def pipeline():
    return ProfileAnthropometricPipeline.__new__(ProfileAnthropometricPipeline)


def analyze(points):
    return pipeline().perform_anthropometric_analysis(points)


def offset(origin, degrees, length=100.0):
    radians = math.radians(degrees)
    return (origin[0] + math.sin(radians) * length, origin[1] + math.cos(radians) * length)


def test_nose_boundaries():
    def nose(dx):
        result = analyze({
            "24": (0.0, 0.0),
            "10": (0.0, 1000.0),
            "18": (0.0, 500.0),
            "17": (dx, 500.0),
        })
        return result["nose_classification"], result["nose_normalized"]

    assert nose(195.0) == ("nariz normal", pytest.approx(0.195))
    assert nose(160.0)[0] == "nariz normal"
    assert nose(195.1)[0] == "nariz protruyente"
    assert nose(159.9)[0] == "nariz corta"


def test_mandible_boundaries_drop_intermediate_bands():
    def jaw(width):
        result = analyze({
            "24": (-100.0, 0.0),
            "10": (0.0, 1000.0),
            "18": (0.0, 0.0),
            "3": (0.0, 500.0),
            "9": (width, 500.0),
        })
        return result["mandibula_classification"], result["mandibula_normalized"]

    assert jaw(775.0) == ("Mandibula Sanguinea", pytest.approx(0.775))
    assert jaw(350.0) == ("Mandibula Bilosa", pytest.approx(0.35))
    assert jaw(774.9)[0] == "Mandibula Bilosa"
    assert jaw(349.9)[0] == "Mandibula Nerviosa"
    assert "intermedia" not in jaw(770.0)[0].lower()
    assert "intermedia" not in jaw(370.0)[0].lower()


def test_ear_length_boundaries():
    def ear(length):
        result = analyze({
            "24": (0.0, 0.0),
            "10": (0.0, 10000.0),
            "4": (0.0, 0.0),
            "5": (length, 0.0),
        })
        return result["ear_length_classification"], result["ear_length_proportion"]

    assert ear(4325.0) == ("oreja normal", pytest.approx(0.4325))
    assert ear(3800.0)[0] == "oreja normal"
    assert ear(4326.0)[0] == "oreja larga"
    assert ear(3799.0)[0] == "oreja corta"


def test_ear_lobe_boundaries():
    def lobe(start_x):
        result = analyze({
            "24": (0.0, 0.0),
            "10": (0.0, 100.0),
            "4": (0.0, 0.0),
            "5": (1000.0, 0.0),
            "3": (start_x, 0.0),
        })
        return result["ear_lobe_classification"], result["ear_lobe_proportion"]

    assert lobe(770.0) == ("lobulo normal", pytest.approx(0.23))
    assert lobe(690.0) == ("lobulo normal", pytest.approx(0.31))
    assert lobe(689.0)[0] == "lobulo grande"
    assert lobe(771.0)[0] == "lobulo chico"


def test_forehead_boundaries_drop_neutral():
    def forehead(degrees):
        result = analyze({
            "24": (0.0, 0.0),
            "10": (0.0, 200.0),
            "18": (0.0, 100.0),
            "33": offset((0.0, 0.0), degrees),
        })
        return result["forehead_classification"], result["forehead_angle"]

    degrees = 22.0
    label, angle = forehead(degrees)
    for _ in range(8):
        if angle <= 22.0:
            break
        degrees -= 1e-4
        label, angle = forehead(degrees)
    assert angle == pytest.approx(22.0, abs=1e-3)
    assert angle <= 22.0
    assert label == "frente vertical"

    degrees = 22.0
    label, angle = forehead(degrees)
    for _ in range(8):
        if angle > 22.0:
            break
        degrees += 1e-4
        label, angle = forehead(degrees)
    assert angle == pytest.approx(22.0, abs=1e-3)
    assert label == "frente inclinada hacia atras"
    assert forehead(15.0)[0] == "frente vertical"
    assert "frente neutra" not in {forehead(12.0)[0], forehead(15.0)[0]}


def test_chin_boundaries():
    def chin(degrees, sign):
        # Point 17 left of 18 selects the left-side sign branch.
        # Positive x on 18→11 is negated; negative x stays positive.
        radians = math.radians(degrees)
        dx = math.sin(radians) * 100.0
        if sign > 0:
            dx = -dx
        point_11 = (dx, 100.0 + math.cos(radians) * 100.0)
        result = analyze({
            "24": (0.0, 0.0),
            "10": (0.0, 200.0),
            "18": (0.0, 100.0),
            "17": (-10.0, 100.0),
            "11": point_11,
        })
        return result["chin_classification"], result["chin_angle"]

    label, angle = chin(4.5, -1)
    assert angle == pytest.approx(-4.5, abs=1e-3)
    assert angle <= -4.5
    assert label == "menton nervioso"

    degrees = 5.5
    label_55, angle_55 = chin(degrees, 1)
    for _ in range(8):
        if angle_55 <= 5.5:
            break
        degrees -= 1e-4
        label_55, angle_55 = chin(degrees, 1)
    assert angle_55 == pytest.approx(5.5, abs=1e-3)
    assert angle_55 <= 5.5
    assert label_55 == "menton biloso/linfatico"

    degrees = 5.5
    label_over, angle_over = chin(degrees, 1)
    for _ in range(8):
        if angle_over > 5.5:
            break
        degrees += 1e-4
        label_over, angle_over = chin(degrees, 1)
    assert angle_over == pytest.approx(5.5, abs=1e-3)
    assert label_over == "menton sanguineo"
    assert chin(3.0, -1)[0] == "menton biloso/linfatico"


def test_eye_protrusion_labels_only_on_left_profile():
    analyzer = pipeline()
    orbital_base = (0.0, 0.0)
    orbital_top = (0.0, 100.0)

    def classify(cornea_x, head_direction):
        return analyzer.calculate_eye_protrusion_classification(
            orbital_base, orbital_top, (cornea_x, 50.0), head_direction
        )

    distance, label = classify(1.0, "left")
    assert distance == pytest.approx(1.0)
    assert label == "protusion positiva"

    distance, label = classify(0.0, "left")
    assert distance == pytest.approx(0.0)
    assert label == "protusion nula"

    distance, label = classify(-3.0, "left")
    assert distance == pytest.approx(-3.0)
    assert label == "protusion nula"

    distance, label = classify(-3.1, "left")
    assert label == "protusion negativa"

    distance, label = classify(-1.0, "right")
    assert distance == pytest.approx(1.0)
    assert label is None

    right_profile = analyze({
        "24": (0.0, 0.0),
        "18": (10.0, 100.0),
        "10": (0.0, 200.0),
        "37": orbital_base,
        "39": orbital_top,
        "38": (-5.0, 50.0),
    })
    assert right_profile["head_direction"] == "right"
    assert "eye_protrusion_distance" in right_profile
    assert "eye_protrusion_classification" not in right_profile
