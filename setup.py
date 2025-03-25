from setuptools import find_packages, setup

package_name = 'reach'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,  # Asegura que se incluyan los archivos listados en package_data
    package_data={
        # Indica que en el paquete 'scripts' se incluya el archivo 'policy.pt'
        'scripts': ['policy.onnx'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='inaki',
    maintainer_email='inakirm111@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reach = scripts.reach:main'
        ],
    },
)
