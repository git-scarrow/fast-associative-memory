"""
shutter_deck/service.py — Main entry point for the shutter-deck-fam systemd service.

Reads config.toml, initializes the encoder + CAM + persistence, installs
density-aware eviction, and starts the inotify ingestion loop.
Handles SIGTERM for graceful shutdown with final state save.
"""

import logging
import signal
import sys
import tomllib
from functools import partial
from pathlib import Path

import torch

logger = logging.getLogger("shutter_deck")


def load_config(config_path: Path) -> dict:
    """Parse config.toml and return the full config dict."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_cam(cfg: dict):
    """Construct a ContinuousCAM from the [cam] config section."""
    from associative_core import ContinuousCAM

    cam_cfg = cfg["cam"]
    return ContinuousCAM(
        key_dim=cam_cfg["key_dim"],
        value_dim=cam_cfg["value_dim"],
        max_entries=cam_cfg["max_entries"],
        vigilance=cam_cfg["vigilance"],
        hebb_lr=cam_cfg["hebb_lr"],
        key_lr=cam_cfg["key_lr"],
        inference_k=cam_cfg["inference_k"],
        inference_temp=cam_cfg["inference_temp"],
        use_lfu=cam_cfg["use_lfu"],
        use_bfloat16=cam_cfg["use_bfloat16"],
        immutable_keys=cam_cfg["immutable_keys"],
    )


def main(config_path: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config_path = Path(config_path or "config.toml")
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    ingest_cfg = cfg["ingest"]

    # Build CAM
    logger.info("Building CAM (max_entries=%d) …", cfg["cam"]["max_entries"])
    cam = build_cam(cfg)

    # Install density-aware eviction
    from shutter_deck.eviction import install_density_eviction
    install_density_eviction(cam, min_per_class=cfg["eviction"]["min_per_class"])
    logger.info("Density-aware eviction installed (floor=%d).", cfg["eviction"]["min_per_class"])

    # Load persisted state
    from shutter_deck.persistence import load_cam_state, save_cam_state
    state_path = Path(paths_cfg["state_file"])
    load_cam_state(cam, state_path)

    # Save helper
    save_fn = partial(save_cam_state, cam, state_path)

    # SIGTERM handler for graceful shutdown
    def _on_sigterm(signum, frame):
        logger.info("SIGTERM received — saving state and exiting.")
        save_fn()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    # Build encoder
    from shutter_deck.encoder import DINOv2Encoder
    enc_cfg = cfg["encoder"]
    encoder = DINOv2Encoder(
        model_name=enc_cfg["model_name"],
        device=enc_cfg.get("device", "cpu"),
        model_path=enc_cfg.get("model_path"),
        backend=enc_cfg.get("backend", "onnxruntime"),
        num_threads=enc_cfg.get("num_threads", 4),
        input_size=enc_cfg.get("input_size", 224),
    )
    encoder.warmup()

    # Open metadata DB
    from shutter_deck.ingest import init_metadata_db, watch_and_ingest
    metadata_conn = init_metadata_db(Path(paths_cfg["metadata_db"]))

    # Start ingestion loop (blocks forever)
    logger.info("Starting ingestion loop.")
    watch_and_ingest(
        cam=cam,
        encoder=encoder,
        watch_dir=Path(paths_cfg["watch_dir"]),
        metadata_conn=metadata_conn,
        save_fn=save_fn,
        batch_size=ingest_cfg["batch_size"],
        batch_window_sec=ingest_cfg["batch_window_sec"],
        checkpoint_interval=ingest_cfg["checkpoint_interval"],
        patterns=tuple(ingest_cfg["file_extensions"]),
        webhook_url=ingest_cfg.get("webhook_url"),
    )


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else None
    main(config)
