from setuptools import setup, find_packages

package_name = 'reach'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={
        # Include policy.onnx as part of the installed package
        package_name: ['policy.onnx'],
    },
    data_files=[
        # Ament index registration
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/inference.launch.py']),
    ],

    install_requires=[
        'setuptools',
        'numpy',
        'onnxruntime',
    ],
    zip_safe=True,
    author='inaki',
    author_email='inakirm111@gmail.com',
    description='Minimal ROS 2 package for ONNX-based policy inference',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            # This exposes the script as an executable
            'policy_inference_node = reach.policy_inference_node:main',
            'goal_position_node = reach.goal_position_node:main',
            'joint_states_node = reach.joint_states_node:main',
            'moveit_node = reach.moveit_node:main',
            'ur5e_moveit_client = reach.ur5e_moveit_client:main',
        ],
    },
)
