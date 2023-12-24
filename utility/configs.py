import importlib.util
import os.path
import pprint
import sys


class Config:
    """
    A class to manage configuration settings for the application.

    The configuration settings can be loaded from a Python file, and the current
    configuration can be saved back to a file. The class allows attribute-style
    access to configuration parameters.

    Attributes:
        Any configuration parameter loaded from the configuration file.

    Methods:
        load_from_file(config_path): Class method to load configuration from a file.
        save_to_file(file_path): Instance method to save current configuration to a file.
    """

    def __init__(self, dictionary=None, config_file_name=None, append_info="", prepend_info="", save=True):
        """
        Initialize the Config object with a dictionary of configuration parameters.

        If a dictionary is provided, the method sets each key-value pair in the
        dictionary as an attribute of the Config object.

        Args:
            dictionary (dict, optional): A dictionary of configuration parameters.
                                         Defaults to None.
        """
        self.fragment_ids = None
        if dictionary:
            for key, value in dictionary.items():
                setattr(self, key, value)
        self.save = save
        self.config_file_name = config_file_name
        self.append_info = append_info
        self.prepend_info = prepend_info

    @classmethod
    def load_local_cfg(cls):
        config = {}

        # Check for and apply local configuration overrides
        local_config_path = '../conf_local.py'
        if os.path.exists(local_config_path):
            local_config = cls.import_config_from_path(local_config_path)
            config.update({k: v for k, v in vars(local_config).items() if not k.startswith('__')})

        return cls(config, None, save=False)

    @classmethod
    def load_from_file(cls, config_path):
        """
        Load configuration from a Python file and return a Config object.

        This class method dynamically imports the specified Python file as a module,
        then reads its configuration parameters, including those from base
        configuration files specified in the '_base_' attribute of the imported module.

        Args:
            config_path (str): The file path of the configuration file to load.

        Returns:
            Config: A Config object initialized with the loaded configuration parameters.
        """

        append_info = ""
        # Read the source file
        with open(config_path, 'r') as file:
            prepend_info = ''.join(line for line in file if line.startswith('import'))
            prepend_info += ''.join(line for line in file if line.startswith('from'))
            prepend_info += ''.join(line for line in file if line.startswith('sys.path'))

        with open(config_path, 'r') as file:
            content = file.read()
            start_index = content.find("train_aug")
            if start_index != -1:
                append_info = content[start_index:]
            else:
                print("Warning! No train_aug found in ", config_path)

        model_config = cls.import_config_from_path(config_path)
        config = {}
        # todo is this the right way to handle this? was necessary to add since my dataset configs don't have base
        if hasattr(model_config, '_base_'):
            for base_path in model_config._base_:
                base_config = cls.import_config_from_path(base_path)
                config.update({k: v for k, v in vars(base_config).items() if not k.startswith('__')})

        config.update({k: v for k, v in vars(model_config).items() if not k.startswith('__')})

        # Check for and apply local configuration overrides
        local_config_path = 'conf_local.py'
        if os.path.exists(local_config_path):
            print("Found local config")
            local_config = cls.import_config_from_path(local_config_path)
            config.update({k: v for k, v in vars(local_config).items() if not k.startswith('__')})

        if config_path.__contains__('/'):
            config_file_name = config_path.split('/')[-1]
        elif config_path.__contains__('\\'):
            config_file_name = config_path.split('\\')[-1]
        else:
            print("Error: Config path incorrect.")
            sys.exit(1)
        return cls(config, config_file_name, append_info=append_info, prepend_info=prepend_info)

    def save_to_file(self, model_run_dir, file_path=None):
        """
        Save the current configuration to a Python file.

        If a file path is provided, the configuration is saved to that path. If no file path
        is provided, the method attempts to save the configuration to the original configuration
        file's path that was used to load this configuration (if available). If neither a file path
        is provided nor an original path is available, a ValueError is raised.

        Each configuration parameter is written to the file as a line in the format:
        'key = value', where 'key' is the name of the configuration parameter, and
        'value' is its value represented as a Python literal.

        Args:
            file_path (str, optional): The file path where the configuration will be saved.
                                       If None, tries to use the original configuration file's path.
                                       Defaults to None.

        Raises:
            ValueError: If both file_path is None and the original configuration file's path is unknown.
        """
        if not self.save:
            print("Only a local config has been loaded, which cannot be saved.")
            sys.exit(1)

        if file_path is None:
            file_path = self.config_file_name
            if file_path is None:
                raise ValueError("Target file path not specified and original config path unknown.")

        keys_to_ignore = [
            'config_file_name',
            'train_aug',
            'val_aug',
            'os',
            'A',
            'prepend_info',
            'append_info'
        ]

        file_path = os.path.join(model_run_dir, self.config_file_name)
        with open(file_path, 'w') as f:
            f.write(f'{self.prepend_info}')
            for key, value in self.__dict__.items():
                if key not in keys_to_ignore:
                    f.write(f'{key} = {repr(value)}\n')
            f.write(f'{self.append_info}')

    def __str__(self):
        """
        Return a string representation of the configuration, in a pretty-printed format.

        Returns:
            str: A formatted string of the configuration dictionary.
        """
        return pprint.pformat(self.__dict__, indent=4, width=1)

    @staticmethod
    def import_config_from_path(path):
        module_name = os.path.basename(path).split('.')[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
