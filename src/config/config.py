import os
import yaml

'''
The RootConfig class is inherited to have some very common parameters
'''


class Config(object):
    # Root directory for entire repo
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Local directories contain logs, datasets, saved model
    data_dir = os.path.join(root_dir, 'local', 'data')
    log_dir = os.path.join(root_dir, 'local', 'logs')
    model_dir = os.path.join(root_dir, 'local', 'models')
    config_dir = os.path.join(root_dir, 'src', 'config')
    # Images dimensions
    image_width = 128
    image_height = 96
    image_channels = 1

    @classmethod
    def from_yaml(cls, yaml_name):
        # If a yaml file is given, unpack into config properties
        if yaml_name is not None:
            yaml_path = os.path.join(cls.config_dir, yaml_name)
            if not os.path.exists(yaml_path):
                raise Exception('Config yaml not found at %s' % yaml_path)
            with open(yaml_path, 'r') as stream:
                try:
                    params = yaml.load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            for key, value in params.items():
                setattr(cls, key, value)
        return cls



