from setuptools import find_packages, setup

package_name = 'sspid2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'sspid2/config/custom_tracker.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hawkeye',
    maintainer_email='workingelk@gmail.com',
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
