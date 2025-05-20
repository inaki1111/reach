from setuptools import setup, find_packages

package_name = 'reach'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'reach': ['policy.onnx', 'policy.pt'],
    },
    data_files=[
        # Marca este paquete en el índice ament
        ('share/ament_index/resource_index/packages', ['resource/reach']),
        # Copia package.xml al install space
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'numpy',
        'pin',
    ],
    zip_safe=True,
    author='inaki',
    author_email='inakirm111@gmail.com',
    description='ROS 2 package for testing Differential IK Controller',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            # Apuntamos al módulo scripts.ik_node
            'ik_node = reach.ik_node:main',
            # Si quieres exponer también tu reach.py
            'reach   = reach.reach:main',
        ],
    },
)
