"""Disconnect opacity-from-texture-alpha on every UsdPreviewSurface material in a USD file.

Blender's FBX/glTF importer auto-wires an image texture's alpha output to the
Principled BSDF's Alpha input whenever the image has an alpha channel, regardless of
the material's Blend Mode. USD export carries that connection through, and IsaacSim/
Omniverse then renders the object fully or partially transparent (unlike Blender's own
viewport, which ignores the link outside Alpha Blend mode). This script removes that
connection and pins opacity to 1.0, leaving base color / other inputs untouched.

Usage (from the project root, with the Isaac Sim venv active):
    source $HOME/isaacsim/env_isaaclab/bin/activate
    export OMNI_KIT_ACCEPT_EULA=YES
    python3 scripts/fix_opacity.py envs/some_model.usd
    python3 scripts/fix_opacity.py envs/some_model.usd --no-backup
    python3 scripts/fix_opacity.py envs/some_model.usd -o envs/some_model_fixed.usd

pxr requires the Kit runtime to be bootstrapped first (plain `python3 -c "import pxr"`
fails with ModuleNotFoundError) -- that's what the isaacsim.SimulationApp call below is
for. It costs a few seconds of startup per run.
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("usd_path", help="Path to the .usd/.usda/.usdc file to fix (edited in place by default)")
    parser.add_argument("-o", "--output", help="Write to this path instead of overwriting usd_path")
    parser.add_argument("--no-backup", action="store_true", help="Skip writing a .bak copy before saving in place")
    args = parser.parse_args()

    usd_path = os.path.abspath(args.usd_path)
    if not os.path.isfile(usd_path):
        print(f"[fix_opacity] file not found: {usd_path}")
        sys.exit(1)

    in_place = args.output is None
    if in_place and not args.no_backup:
        backup_path = usd_path + ".bak"
        shutil.copyfile(usd_path, backup_path)
        print(f"[fix_opacity] backup written: {backup_path}")

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": True})

    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(usd_path)

    total_preview_surface = 0
    fixed = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        total_preview_surface += 1
        opacity_input = shader.GetInput("opacity")
        if opacity_input is None:
            continue
        conn = opacity_input.GetConnectedSources()
        if conn and conn[0]:
            opacity_input.DisconnectSource()
            opacity_input.Set(1.0)
            fixed += 1

    out_path = usd_path if in_place else os.path.abspath(args.output)
    if in_place:
        stage.GetRootLayer().Save()
    else:
        stage.GetRootLayer().Export(out_path)

    print(f"[fix_opacity] UsdPreviewSurface shaders: {total_preview_surface}", flush=True)
    print(f"[fix_opacity] opacity connections removed: {fixed}", flush=True)
    print(f"[fix_opacity] saved: {out_path}", flush=True)

    # SimulationApp.close() tears the process down hard enough that a block-buffered
    # stdout (the normal case once stdout isn't a tty, e.g. piped/redirected) can lose
    # everything printed above if it isn't flushed first.
    sys.stdout.flush()
    sim_app.close()


if __name__ == "__main__":
    main()
