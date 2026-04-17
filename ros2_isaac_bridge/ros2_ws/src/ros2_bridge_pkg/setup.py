from setuptools import find_packages, setup

package_name = 'ros2_bridge_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/competition.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ammar',
    maintainer_email='ammar.issa.500@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bridge_node = ros2_bridge_pkg.bridge_node:main',
            'controller_node = ros2_bridge_pkg.controller_node:main',
            'dashboard_node = ros2_bridge_pkg.dashboard_node:main',
        ],
    },
)
