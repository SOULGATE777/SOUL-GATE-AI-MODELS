"""Unit tests for Photoroom client (no live HTTP)."""
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app.utils import photoroom_client


def test_is_configured_false_when_missing(monkeypatch):
    monkeypatch.delenv("PHOTOROOM_API_KEY", raising=False)
    assert photoroom_client.is_configured() is False


def test_is_configured_true_when_set(monkeypatch):
    monkeypatch.setenv("PHOTOROOM_API_KEY", "test-key")
    assert photoroom_client.is_configured() is True


def test_remove_background_white_missing_key_returns_none(monkeypatch):
    monkeypatch.delenv("PHOTOROOM_API_KEY", raising=False)
    image = np.full((8, 8, 3), 40, dtype=np.uint8)
    assert photoroom_client.remove_background_white(image) is None


def test_remove_background_white_rejects_non_allowlisted_url(monkeypatch):
    monkeypatch.setenv("PHOTOROOM_API_KEY", "test-key")
    monkeypatch.setenv("PHOTOROOM_API_URL", "https://evil.example/v1/segment")
    image = np.full((8, 8, 3), 40, dtype=np.uint8)
    with patch.object(photoroom_client.requests, "post") as post:
        assert photoroom_client.remove_background_white(image) is None
        post.assert_not_called()


def test_remove_background_white_http_error_fail_open(monkeypatch):
    monkeypatch.setenv("PHOTOROOM_API_KEY", "test-key")
    monkeypatch.delenv("PHOTOROOM_API_URL", raising=False)
    image = np.full((8, 8, 3), 40, dtype=np.uint8)
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.content = b""
    with patch.object(photoroom_client.requests, "post", return_value=mock_resp) as post:
        assert photoroom_client.remove_background_white(image) is None
        post.assert_called_once()
        kwargs = post.call_args.kwargs
        assert kwargs["headers"]["x-api-key"] == "test-key"
        assert kwargs["timeout"] == photoroom_client.REQUEST_TIMEOUT_SEC
        assert kwargs["verify"] is True
        assert kwargs["allow_redirects"] is False
        assert kwargs["data"]["size"] == "full"


def test_remove_background_white_success_decodes_png(monkeypatch):
    monkeypatch.setenv("PHOTOROOM_API_KEY", "test-key")
    monkeypatch.delenv("PHOTOROOM_API_URL", raising=False)
    image = np.full((10, 12, 3), 40, dtype=np.uint8)

    white_bgr = np.full((10, 12, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", white_bgr)
    assert ok

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = buf.tobytes()

    with patch.object(photoroom_client.requests, "post", return_value=mock_resp):
        out = photoroom_client.remove_background_white(image)

    assert out is not None
    assert out.shape == (10, 12, 3)
    assert np.all(out == 255)


def test_remove_background_white_resizes_mismatched_output(monkeypatch):
    monkeypatch.setenv("PHOTOROOM_API_KEY", "test-key")
    monkeypatch.delenv("PHOTOROOM_API_URL", raising=False)
    image = np.full((20, 30, 3), 40, dtype=np.uint8)

    # Photoroom returns a different resolution; client must resize to input HxW.
    small_bgr = np.full((10, 15, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", small_bgr)
    assert ok

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = buf.tobytes()

    with patch.object(photoroom_client.requests, "post", return_value=mock_resp):
        out = photoroom_client.remove_background_white(image)

    assert out is not None
    assert out.shape == (20, 30, 3)


def test_normalize_photoroom_bg_color_allowlist():
    assert photoroom_client.normalize_photoroom_bg_color(None) == "white"
    assert photoroom_client.normalize_photoroom_bg_color("") == "white"
    assert photoroom_client.normalize_photoroom_bg_color("WHITE") == "white"
    assert photoroom_client.normalize_photoroom_bg_color("#FFFFFF") == "white"
    assert photoroom_client.normalize_photoroom_bg_color("#a6a6a6") == "#a6a6a6"
    assert photoroom_client.normalize_photoroom_bg_color("#A6A6A6") == "#a6a6a6"
    assert photoroom_client.normalize_photoroom_bg_color("red") == "white"
    assert photoroom_client.normalize_photoroom_bg_color("#ff0000") == "white"


def test_bg_color_to_rgb():
    assert photoroom_client.bg_color_to_rgb("white") == (255, 255, 255)
    assert photoroom_client.bg_color_to_rgb("#a6a6a6") == (166, 166, 166)
    assert photoroom_client.bg_color_to_rgb("nope") == (255, 255, 255)


def test_remove_background_white_posts_gray_bg_color(monkeypatch):
    monkeypatch.setenv("PHOTOROOM_API_KEY", "test-key")
    monkeypatch.delenv("PHOTOROOM_API_URL", raising=False)
    image = np.full((8, 8, 3), 40, dtype=np.uint8)
    gray_bgr = np.full((8, 8, 3), 166, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", gray_bgr)
    assert ok
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = buf.tobytes()
    with patch.object(photoroom_client.requests, "post", return_value=mock_resp) as post:
        out = photoroom_client.remove_background_white(image, bg_color="#a6a6a6")
    assert out is not None
    assert post.call_args.kwargs["data"]["bg_color"] == "#a6a6a6"
