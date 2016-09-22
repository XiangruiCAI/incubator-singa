from google.protobuf import text_format
import argparse
import re
import sys

caffe_flag = True
try:
    import caffe
    from caffe.proto import caffe_pb2
except ImportError:
    caffe_flag = False
    import caffe_parse.caffe_pb2

# Set configuration of solver
# Read caffe proto file
def readProtoSolverFile(filepath):
    solver_config = ''
    if caffe_flag:
        solver_config = caffe.proto.caffe_pb2.NetParameter()
    else:
        solver_config = caffe_parse.caffe_pb2.NetParameter()
    return readProtoFile(filepath, solver_config)

# Read and parse caffe proto file
def readProtoFile(filepath, parser_object):
    file = open(filepath, "r")
    if not file:
        raise self.ProcessException("ERROR (" + filepath + ")!")
    # Merges an ASCII representation of a protocol message into a message.
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object

# Convert layer parameters into a string,
# including num_filter, pad, kernel, stride, bias
# Supposing heights and widths of the parameters are always the same
def convParamToString(param):
    pad = 0
    if isinstance(param.pad, int):
        pad = param.pad
    else:
        pad = 0 if len(param.pad) == 0 else param.pad[0]
    stride = 1
    if isinstance(param.stride, int):
        stride = param.stride
    else:
        stride = 1 if len(param.stride) == 0 else param.stride[0]
    kernel_size = ''
    if isinstance(param.kernel_size, int):
        kernel_size = param.kernel_size
    else:
        kernel_size = param.kernel_size[0]
    dilate = 1
    if isinstance(param.dilation, int):
        dilate = param.dilation
    else:
        dilate = 1 if len(param.dilation) == 0 else param.dilation[0]
    # convert to string except for dilation
    param_string = "num_filter=%d, pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d), no_bias=%s" %\
        (param.num_output, pad, pad, kernel_size,\
        kernel_size, stride, stride, not param.bias_term)
    # deal with dilation. Won't be in deconvolution
    if dilate > 1:
        param_string += ", dilate=(%d, %d)" % (dilate, dilate)
    return param_string

# Convert caffe model into a singa model
def proto2script(proto_file):
    proto = readProtoSolverFile(proto_file)
    connection = dict()
    symbols = dict()
    top = dict()
    flatten_count = 0
    symbol_string = ""
    layer = ''
    if len(proto.layer):
        layer = proto.layer
    elif len(proto.layers):
        layer = proto.layers
    else:
        raise Exception('Invalid proto file.')
    # Get input size to network
    input_dim = [1, 3, 224, 224] # default
    input_dim = layer[0].input_param.shape.dim[0]
    print layer[0].input_param.shape.dim
    print input_dim
    # We assume the first bottom blob of first layer is the output from data layer
    print "layer[0]:\n", layer[0]
    if hasattr(layer[0], bottom):
        print layer[0].bottom[0]
    else:
        print "hello"

    # layer[0] may have no bottom, why did they call this attribute?
    input_name = layer[0].bottom[0]
    output_name = ""
    mapping = {input_name : 'data'}
    need_flatten = {input_name : False}
    # I should know the structure of SIGNA layer classes.
    for i in range(len(layer)):
        type_string = ''
        param_string = ''
        name = re.sub('[-/]', '_', layer[i].name)
        if layer[i].type == 'Convolution' or layer[i].type == 4:
            type_string = 'singa.layer.Conv2D'
            param_string = convParamToString(layer[i].convolution_param)
            need_flatten[name] = True
        #if layer[i].type == 'Deconvolution' or layer[i].type == 39:
        #    type_string = 'mx.symbol.Deconvolution'
        #    param_string = convParamToString(layer[i].convolution_param)
        #    need_flatten[name] = True
        if layer[i].type == 'Pooling' or layer[i].type == 17:
            type_string = 'singa.layer.Pooling2D'
            param = layer[i].pooling_param
            param_string = ''
            if param.global_pooling == True:
                # there must be a param `kernel` in a pooling layer
                param_string += "global_pool=True, kernel=(1,1)"
            else:
                param_string += "pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d)" %\
                    (param.pad, param.pad, param.kernel_size,\
                    param.kernel_size, param.stride, param.stride)
            if param.pool == 0:
                param_string = param_string + ", mode=model_pb2.PoolingConf.MAX"
            elif param.pool == 1:
                param_string = param_string + ", mode=model_pb2.PoolingConf.AVE"
            else:
                raise Exception("Unknown Pooling Method!")
            need_flatten[name] = True
        if layer[i].type == 'ReLU' or layer[i].type == 18:
            type_string = 'singa.layer.Activation'
            param_string = "mode='relu'"
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'LRN' or layer[i].type == 15:
            type_string = 'singa.layer.LRN'
            param = layer[i].lrn_param
            param_string = "size=%d, alpha=%f, beta=%f, k=%f" %\
                (param.local_size, param.alpha, param.beta, param.k)
            need_flatten[name] = True
        if layer[i].type == 'InnerProduct' or layer[i].type == 14:
            type_string = 'singa.layer.Dense'
            param = layer[i].inner_product_param
            param_string = "num_output=%d, use_bias=%s" % (param.num_output, param.bias_term)
            need_flatten[name] = False
        if layer[i].type == 'Dropout' or layer[i].type == 6:
            type_string = 'singa.layer.Dropout'
            param = layer[i].dropout_param
            param_string = "p=%f" % param.dropout_ratio
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'Softmax' or layer[i].type == 20:
            type_string = 'singa.layer.Softmax'
        if layer[i].type == 'Flatten' or layer[i].type == 8:
            type_string = 'singa.layer.Flatten'
            need_flatten[name] = False
        # which model will use Split layer?
        if layer[i].type == 'Split' or layer[i].type == 22:
            type_string = 'singa.layer.Split'
        # no concat layer in singa?
        if layer[i].type == 'Concat' or layer[i].type == 3:
            type_string = 'mx.symbol.Concat'
            need_flatten[name] = True
        # ??Do we have Crop layer?
        if layer[i].type == 'Crop':
            type_string = 'mx.symbol.Crop'
            need_flatten[name] = True
            param_string = 'center_crop=True'
        if layer[i].type == 'BatchNorm':
            type_string = 'singa.layer.BatchNormalization'
            param = layer[i].batch_norm_param
        if type_string == '':
            raise Exception('Unknown Layer %s!' % layer[i].type)
        if type_string != 'split':
            bottom = layer[i].bottom
            if param_string != "":
                param_string = ", " + param_string
            if len(bottom) == 1:
                if need_flatten[mapping[bottom[0]]] and type_string == 'mx.symbol.FullyConnected':
                    flatten_name = "flatten_%d" % flatten_count
                    symbol_string += "%s=mx.symbol.Flatten(name='%s', data=%s)\n" %\
                        (flatten_name, flatten_name, mapping[bottom[0]])
                    flatten_count += 1
                    need_flatten[flatten_name] = False
                    bottom[0] = flatten_name
                    mapping[bottom[0]] = bottom[0]
                symbol_string += "%s = %s(name='%s', data=%s %s)\n" %\
                    (name, type_string, name, mapping[bottom[0]], param_string)
            else:
                symbol_string += "%s = %s(name='%s', *[%s] %s)\n" %\
                    (name, type_string, name, ','.join([mapping[x] for x in bottom]), param_string)
        for j in range(len(layer[i].top)):
            mapping[layer[i].top[j]] = name
        output_name = name
    return symbol_string, output_name, input_dim

def proto2symbol(proto_file):
    sym, output_name, input_dim = proto2script(proto_file)
    sym = "import mxnet as mx\n" \
            + "data = mx.symbol.Variable(name='data')\n" \
            + sym
    exec(sym)
    exec("ret = " + output_name)
    return ret, input_dim

def main():
    print sys.argv[0]
    print sys.argv[1]
    symbol_string, output_name = proto2script(sys.argv[1])
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as fout:
            fout.write(symbol_string)
    else:
        print(symbol_string)

if __name__ == '__main__':
    main()
