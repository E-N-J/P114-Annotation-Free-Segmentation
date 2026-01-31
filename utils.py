import sys

def get_environment():
    """
    Returns a string indicating the environment: 'colab', 'notebook', or 'script'.
    """
    if 'google.colab' in sys.modules:
        return 'colab'
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'notebook'
        if 'terminal' in ipy_str:
            return 'script'
    except NameError:
        # get_ipython() is not defined, so we are in standard Python
        return 'script'
        
    return 'script'
