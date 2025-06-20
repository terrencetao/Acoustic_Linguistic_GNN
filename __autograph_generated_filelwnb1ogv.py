# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf___jit_compiled_convolution_op(self, inputs, kernel):
            with ag__.FunctionScope('_jit_compiled_convolution_op', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(self).convolution_op, (ag__.ld(inputs), ag__.ld(kernel)), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___jit_compiled_convolution_op
    return inner_factory