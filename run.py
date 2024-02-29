import importlib.util
import os


def hello():
    print("Hello! This is the Vershachi Unlearning Framework.")


def read_requirements(file_path):
    with open(file_path, "r") as f:
        requirements = [line.strip() for line in f.readlines() if line.strip()]
    return requirements


def check_module(module_name):
    try:
        importlib.util.find_spec(module_name)
        print(f"Module '{module_name}' is installed.")
        return True
    except ImportError:
        print(f"Module '{module_name}' is not installed.")
        return False


def sanity_checks():
    print("Performing sanity checks...")
    dev_requirements = read_requirements("requirements-dev.txt")
    requirements = read_requirements("requirements.txt")
    all_requirements = set(dev_requirements + requirements)
    all_modules_installed = all(check_module(module) for module in all_requirements)
    if all_modules_installed:
        print("All required modules are installed.")
    else:
        print(
            "Some required modules are missing. Please install them before running the framework."
        )


def version():
    # Display version information
    print("Vershachi Unlearning Framework version 0.1")


if __name__ == "__main__":
    hello()
    sanity_checks()
    version()
