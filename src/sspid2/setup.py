from glob import glob

from setuptools import find_packages, setup

package_name = 'sspid2'

# The perception node loads its configuration from
# get_package_share_directory('sspid2')/config, i.e. share/sspid2/config.
# The YOLO weights are installed from the same place as custom_tracker.yaml so
# that no manual copy into the install tree is needed. The glob keeps the build
# working when the weights have not been downloaded yet; in that case
# distance_node will raise a clear FileNotFoundError at start-up.
config_files = ['sspid2/config/custom_tracker.yaml'] + sorted(glob('sspid2/config/*.pt'))

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', config_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mario Lazzarini Sola',
    maintainer_email='mario.lazzarinisola@study.thws.de',
    description='Vision-based UAV gimbal tracking with SS-PD control and NPSO tuning.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'distance_node = sspid2.distance_to_center:main',
            'controller_node = sspid2.controller:main',
        ],
    },
)
