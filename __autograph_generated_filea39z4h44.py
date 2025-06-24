# coding=utf-8
def outer_factory():
    extract_fn = None

    def inner_factory(ag__):
        tf__lam = lambda x, y: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.py_function, (extract_fn, [x, y], [ag__.converted_call(tf.TensorSpec, (), dict(shape=(None, 768), dtype=tf.float32), lscope), y.dtype]), None, lscope), 'lscope', ag__.STD)
        return tf__lam
    return inner_factory