import re

path = "omniverse_sim.py"
with open(path) as f:
    src = f.read()

new_func = '''def setup_custom_env():
    if args_cli.terrain != 'flat':
        return
    usd_path = f"./envs/{args_cli.custom_env}.usd"
    if not os.path.isfile(usd_path):
        print(f"[go2_omniverse] custom env usd not found: {usd_path}")
        return
    try:
        cfg_scene = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg_scene.func(f"/World/{args_cli.custom_env}", cfg_scene, translation=(0.0, 0.0, 0.0))

        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(f"/World/{args_cli.custom_env}")
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
    except Exception as e:
        print(f"[go2_omniverse] Error loading custom environment '{args_cli.custom_env}': {e}")
'''

pattern = re.compile(r"def setup_custom_env\(\):.*?(?=\ndef capture_hero_shots)", re.S)
m = pattern.search(src)
if not m:
    raise SystemExit("setup_custom_env() が見つかりませんでした。手動確認してください。")

src2 = pattern.sub(new_func + "\n", src, count=1)

with open(path, "w") as f:
    f.write(src2)

print("setup_custom_env() を置き換えました")
