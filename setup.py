from setuptools import find_packages, setup

package_name = 'gs_feature_extraction'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = gs_feature_extraction.publisher_member_function:main',
            'listener = gs_feature_extraction.subscriber_member_function:main',
            'image = gs_feature_extraction.rgb_gray:main',            
            'optical_flow1 = gs_feature_extraction.optical_flow:main',  
            'save_img = gs_feature_extraction.save_images:main',
            'filter_depth = gs_feature_extraction.filter_depth:main',
            'elipse = gs_feature_extraction.elipse_descriptor:main',
            'corners = gs_feature_extraction.corner_detection:main',
            'nn_detect = gs_feature_extraction.detect:main',
            
        ],
    },
)
