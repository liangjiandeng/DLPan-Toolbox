# from .models.builder import PANSHARPENING_MODELS, build_model
from importlib import import_module
import os
join = os.path.join
dirname = os.path.dirname

pkg_list = [import_module('.' + pkg.replace('.py', ''), package="UDL.pansharpening.configs")
            for pkg in os.listdir(dirname(__file__)) if '.py' in pkg]
del pkg_list