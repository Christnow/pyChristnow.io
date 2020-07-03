### 类和函数增加说明装饰器
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        note = """
        Description::
        """
        fn.__doc__ = note + "".join(docstr) + (fn.__doc__ if
                                               fn.__doc__ is not None else "")
        return fn
