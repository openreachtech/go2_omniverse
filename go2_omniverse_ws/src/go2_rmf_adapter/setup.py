from setuptools import find_packages, setup

package_name = "go2_rmf_adapter"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/nav_graph.yaml", "config/fleet_config.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="openreach",
    maintainer_email="noreply@openreach.tech",
    description=(
        "Open-RMF fleet adapter for go2_omniverse's simulated Go2 fleet — see package.xml."
    ),
    license="BSD-2-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fleet_adapter = go2_rmf_adapter.fleet_adapter:main",
        ],
    },
)
