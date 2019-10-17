"""# Measure Execution time

"""
import time


def execution_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time of {original_fn.__name__} : {end_time - start_time}")
        return result

    return wrapper_fn
