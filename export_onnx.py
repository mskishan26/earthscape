"""
Pull a model from the W&B registry and export it to ONNX format.

Fully self-contained: fetches model weights AND source code from W&B artifacts
so there is no dependency on the local models/ directory. This prevents version
mismatches between the code that trained the model and the code used to export it.

Usage:
    # Export from a W&B registry artifact
    uv run python export_onnx.py export \
        --artifact "entity/wandb-registry-earthscape/convnext-models:v0" \
        --output model.onnx

    # Use the final epoch instead of best checkpoint
    uv run python export_onnx.py export \
        --artifact "entity/wandb-registry-earthscape/convnext-models:v0" \
        --checkpoint-name final_model.pth \
        --output model_final.onnx

    # Run inference with the exported ONNX model
    uv run python export_onnx.py infer model.onnx
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ============================================================================
# 1. Resolve W&B artifacts (registry → run ID → model + source code)
# ============================================================================


def resolve_artifacts(artifact_path: str, download_dir: str = ".wandb_artifacts"):
    """
    Given a W&B registry artifact path, trace back to the producer run
    and download both the model artifact and source-code artifact.

    Args:
        artifact_path: W&B artifact reference, e.g.
            "entity/wandb-registry-earthscape/convnext-models:v0"
        download_dir: Local directory for all downloads.

    Returns:
        model_dir: Path to downloaded model artifact (contains .pth files)
        source_dir: Path to downloaded source-code artifact
        run_id: The W&B run ID that produced this model
    """
    import wandb

    api = wandb.Api()

    # Fetch the registry artifact and trace to producer run
    print(f"[W&B] Fetching registry artifact: {artifact_path}")
    artifact = api.artifact(artifact_path)
    run = artifact.logged_by()
    if run is None:
        print("[Error] Could not trace artifact back to a producer run.")
        print("        Make sure the artifact was logged by a W&B run.")
        sys.exit(1)

    run_id = run.id
    project_path = f"{run.entity}/{run.project}"
    print(f"[W&B] Producer run: {run_id} ({project_path})")

    # Download model artifact (model-{run_id})
    model_artifact_name = f"{project_path}/model-{run_id}:latest"
    print(f"[W&B] Fetching model artifact: {model_artifact_name}")
    model_artifact = api.artifact(model_artifact_name, type="model")
    model_dir = model_artifact.download(root=os.path.join(download_dir, f"model-{run_id}"))
    print(f"[W&B] Model weights: {model_dir}")

    # Download source-code artifact (source-{run_id})
    source_artifact_name = f"{project_path}/source-{run_id}:latest"
    print(f"[W&B] Fetching source artifact: {source_artifact_name}")
    source_artifact = api.artifact(source_artifact_name, type="source-code")
    source_dir = source_artifact.download(root=os.path.join(download_dir, f"source-{run_id}"))
    print(f"[W&B] Source code: {source_dir}")

    return model_dir, source_dir, run_id


# ============================================================================
# 2. Dynamic model loading from artifact source code
# ============================================================================


def _dynamic_import_class(module_path: str, class_name: str):
    """Dynamically import a class from a .py file path."""
    spec = importlib.util.spec_from_file_location("_artifact_model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        available = [n for n in dir(module) if not n.startswith("_")]
        print(f"[Error] Class '{class_name}' not found in {module_path}")
        print(f"        Available: {available}")
        sys.exit(1)
    return getattr(module, class_name)


def load_model_from_artifacts(
    model_dir: str,
    source_dir: str,
    checkpoint_name: str = "best.pth",
    device: str = "cpu",
):
    """
    Rebuild a PyTorch model using ONLY the W&B artifacts (no local models/).

    Steps:
        1. Load checkpoint from model artifact → get config + model_meta
        2. Load experiment YAML from source artifact → get features/mode
        3. Dynamically import the model class from the source artifact's model .py
        4. Instantiate model, load weights

    Returns:
        model: nn.Module in eval mode
        cfg: config dict from the checkpoint
        mode: "full" or "rgb"
    """
    # --- Find checkpoint ---
    ckpt_path = os.path.join(model_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        pth_files = sorted(Path(model_dir).glob("*.pth"))
        if pth_files:
            ckpt_path = str(pth_files[0])
            print(f"[Info] '{checkpoint_name}' not found, using: {os.path.basename(ckpt_path)}")
        else:
            print(f"[Error] No .pth files found in {model_dir}")
            sys.exit(1)

    print(f"[Model] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # --- Resolve model class name and source file ---
    model_meta = ckpt.get("model_meta", {})
    class_name = model_meta.get("model_class")
    source_file = model_meta.get("model_source_file")
    mode = model_meta.get("model_mode")
    extra_kwargs = model_meta.get("model_kwargs", {})

    if not class_name or not source_file:
        # Fallback for old checkpoints without model_meta:
        # Find the single .py file in the source artifact's models/ directory
        # (excluding __init__.py) and parse the class from it
        models_dir = os.path.join(source_dir, "models")
        model_files = [
            f for f in os.listdir(models_dir)
            if f.endswith(".py") and f != "__init__.py"
        ]
        if len(model_files) != 1:
            print(f"[Error] Expected 1 model file in artifact, found {len(model_files)}: {model_files}")
            print("        Checkpoint is missing model_meta. Cannot determine which class to use.")
            sys.exit(1)

        source_file = model_files[0]
        print(f"[Fallback] Old checkpoint without model_meta, using: {source_file}")

        # Parse nn.Module subclasses from the file to find the model class
        import ast
        model_py_path = os.path.join(models_dir, source_file)
        with open(model_py_path) as f:
            tree = ast.parse(f.read())
        nn_classes = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                (isinstance(b, ast.Attribute) and b.attr == "Module")
                for b in node.bases
            )
        ]
        if len(nn_classes) == 1:
            class_name = nn_classes[0]
        else:
            print(f"[Error] Found {len(nn_classes)} nn.Module classes in {source_file}: {nn_classes}")
            print("        Cannot determine which class to use without model_meta in checkpoint.")
            sys.exit(1)

    # --- Read experiment YAML from source artifact ---
    exp_dir = os.path.join(source_dir, "experiments")
    if os.path.isdir(exp_dir):
        exp_files = [f for f in os.listdir(exp_dir) if f.endswith(".yaml")]
        if exp_files:
            exp_path = os.path.join(exp_dir, exp_files[0])
            with open(exp_path) as f:
                exp_cfg = yaml.safe_load(f)
            print(f"[Config] Experiment: {exp_cfg.get('name', exp_files[0])}")

            # Use experiment features if not in checkpoint config
            if "_features" not in cfg or not cfg["_features"]:
                features = exp_cfg.get("features", {})
                cfg["_features"] = features
    
    # Determine mode from features if not in model_meta
    if not mode:
        features = cfg.get("_features", {})
        mode = features.get("mode", "full")

    # --- Dynamically import and instantiate the model class ---
    model_py_path = os.path.join(source_dir, "models", source_file)
    print(f"[Model] Importing {class_name} from artifact: models/{source_file}")
    model_cls = _dynamic_import_class(model_py_path, class_name)

    # Build constructor args from config
    features = cfg.get("_features", {})
    model_cfg = cfg["model"]

    if mode == "full":
        spectral_ch = len(features.get("spectral_modalities", []))
        topo_ch = len(features.get("topo_modalities", []))
        model = model_cls(
            num_classes=model_cfg["num_classes"],
            spectral_in_ch=spectral_ch,
            topo_in_ch=topo_ch,
            dropout=model_cfg.get("dropout", 0.3),
            **extra_kwargs,
        )
    elif mode == "rgb":
        rgb_mods = features.get("rgb_modalities", ["aerialr", "aerialg", "aerialb"])
        in_ch = len(rgb_mods)
        model = model_cls(
            num_classes=model_cfg["num_classes"],
            in_channels=in_ch,
            dropout=model_cfg.get("dropout", 0.3),
            **extra_kwargs,
        )
    else:
        print(f"[Error] Unknown mode '{mode}'")
        sys.exit(1)

    # Optional model params
    if "drop_path_rate" in model_cfg:
        # Already handled by constructor if supported; log for visibility
        print(f"[Model] drop_path_rate: {model_cfg['drop_path_rate']}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    arch = model_cfg.get("architecture", class_name)
    print(f"[Model] {arch} → {class_name} ({mode} mode): {param_count:.1f}M params, epoch {ckpt.get('epoch', '?')}")

    return model, cfg, mode


# ============================================================================
# 3. Export to ONNX
# ============================================================================


def export_to_onnx(
    model: torch.nn.Module,
    cfg: dict,
    mode: str,
    output_path: str,
    img_size: int = 256,
    opset_version: int = 18,
):
    """
    Export the model to ONNX format.

    Handles both "full" (spectral + topo) and "rgb" (single tensor) modes.
    """
    features = cfg["_features"]

    if mode == "full":
        spectral_ch = len(features["spectral_modalities"])
        topo_ch = len(features["topo_modalities"])

        dummy_spectral = torch.randn(1, spectral_ch, img_size, img_size)
        dummy_topo = torch.randn(1, topo_ch, img_size, img_size)
        dummy_inputs = (dummy_spectral, dummy_topo)

        input_names = ["spectral", "topo"]
        dynamic_axes = {
            "spectral": {0: "batch_size"},
            "topo": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }
    else:
        rgb_mods = features.get("rgb_modalities", ["aerialr", "aerialg", "aerialb"])
        in_ch = len(rgb_mods)

        dummy_inputs = (torch.randn(1, in_ch, img_size, img_size),)
        input_names = ["image"]
        dynamic_axes = {
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    output_names = ["logits"]

    print(f"[ONNX] Exporting ({mode} mode, opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter to produce a single self-contained file
    )

    # If the dynamo exporter wrote external data (model.onnx.data), merge it
    # back into a single file so we only ship one artifact.
    external_data_path = output_path + ".data"
    if os.path.exists(external_data_path):
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data

        print("[ONNX] Merging external weights into single file...")
        onnx_model = onnx.load(output_path, load_external_data=True)
        # Clear external data references so everything is inline
        for tensor in onnx_model.graph.initializer:
            tensor.ClearField("data_location")
            tensor.ClearField("external_data")
        onnx.save(onnx_model, output_path)
        os.remove(external_data_path)
        print(f"[ONNX] Removed external data file: {external_data_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[ONNX] Saved: {output_path} ({size_mb:.1f} MB)")

    # Validate
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[ONNX] Model validation passed")

    return output_path


# ============================================================================
# 4. ONNX Runtime inference wrapper
# ============================================================================


class ONNXInferenceSession:
    """
    Lightweight ONNX Runtime inference wrapper for EarthScape models.

    Usage:
        session = ONNXInferenceSession("model.onnx")

        # Full mode (spectral + topo)
        logits = session.predict(spectral=spec_array, topo=topo_array)

        # RGB mode
        logits = session.predict(image=rgb_array)

        # Apply sigmoid for multi-label probabilities
        probs = session.predict_proba(spectral=spec_array, topo=topo_array)
    """

    LABEL_COLS = ["af1", "Qal", "Qaf", "Qat", "Qc", "Qca", "Qr"]

    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        import onnxruntime as ort

        if providers is None:
            providers = ort.get_available_providers()

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"[ORT] Loaded: {onnx_path}")
        print(f"[ORT] Inputs:  {self.input_names}")
        print(f"[ORT] Outputs: {self.output_names}")
        print(f"[ORT] Providers: {self.session.get_providers()}")

    @property
    def mode(self) -> str:
        return "full" if "spectral" in self.input_names else "rgb"

    def predict(self, **kwargs) -> np.ndarray:
        """
        Run inference. Pass inputs as keyword arguments matching ONNX input names.

        Args:
            For full mode:  spectral=np.ndarray [B, C_s, H, W], topo=np.ndarray [B, C_t, H, W]
            For rgb mode:   image=np.ndarray [B, C, H, W]

        Returns:
            logits: np.ndarray [B, num_classes]
        """
        feed = {}
        for name in self.input_names:
            if name not in kwargs:
                raise ValueError(
                    f"Missing input '{name}'. Expected: {self.input_names}"
                )
            val = kwargs[name]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            feed[name] = val.astype(np.float32)

        outputs = self.session.run(self.output_names, feed)
        return outputs[0]  # logits

    def predict_proba(self, **kwargs) -> np.ndarray:
        """Run inference and apply sigmoid to get multi-label probabilities."""
        logits = self.predict(**kwargs)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_labels(self, threshold: float = 0.5, **kwargs) -> np.ndarray:
        """Run inference and return binary predictions."""
        probs = self.predict_proba(**kwargs)
        return (probs >= threshold).astype(int)


# ============================================================================
# 5. CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Pull model from W&B registry and export to ONNX"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- Export command ---
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument(
        "--artifact",
        type=str,
        required=True,
        help='W&B registry artifact path, e.g. '
        '"entity/wandb-registry-earthscape/convnext-models:v0"',
    )
    export_parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best.pth",
        help="Checkpoint filename inside the model artifact (default: best.pth)",
    )
    export_parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output ONNX file path (default: model.onnx)",
    )
    export_parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Input image size (default: 256)",
    )
    export_parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    export_parser.add_argument(
        "--download-dir",
        type=str,
        default=".wandb_artifacts",
        help="Directory to download W&B artifacts into (default: .wandb_artifacts)",
    )

    # --- Inference command ---
    infer_parser = subparsers.add_parser("infer", help="Run inference with ONNX model")
    infer_parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the ONNX model file",
    )
    infer_parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Input image size for dummy inference (default: 256)",
    )

    args = parser.parse_args()

    if args.command == "export":
        # Resolve all artifacts from registry → run ID → model + source
        model_dir, source_dir, run_id = resolve_artifacts(
            args.artifact, args.download_dir
        )

        # Load model using artifact source code (no local models/ dependency)
        model, cfg, mode = load_model_from_artifacts(
            model_dir, source_dir, args.checkpoint_name
        )

        # Export to ONNX
        export_to_onnx(model, cfg, mode, args.output, args.img_size, args.opset)

        print(f"\n{'=' * 60}")
        print(f"Export complete! (run: {run_id})")
        print(f"  ONNX model: {args.output}")
        print(f"  Mode: {mode}")
        print(f"  Artifacts: {args.download_dir}")
        print(f"  Verify: uv run python export_onnx.py infer {args.output}")
        print(f"{'=' * 60}")

    elif args.command == "infer":
        # Demo inference with random data
        session = ONNXInferenceSession(args.onnx_path)
        img_size = args.img_size

        if session.mode == "full":
            # Get input shapes from ONNX model
            inputs = session.session.get_inputs()
            spec_ch = inputs[0].shape[1]
            topo_ch = inputs[1].shape[1]
            dummy_spec = np.random.randn(1, spec_ch, img_size, img_size).astype(np.float32)
            dummy_topo = np.random.randn(1, topo_ch, img_size, img_size).astype(np.float32)
            probs = session.predict_proba(spectral=dummy_spec, topo=dummy_topo)
        else:
            inputs = session.session.get_inputs()
            in_ch = inputs[0].shape[1]
            dummy_img = np.random.randn(1, in_ch, img_size, img_size).astype(np.float32)
            probs = session.predict_proba(image=dummy_img)

        print(f"\n[Demo] Probabilities (random input):")
        for name, p in zip(session.LABEL_COLS, probs[0]):
            print(f"  {name}: {p:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
