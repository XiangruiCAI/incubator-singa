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
def readProtoNetFile(filepath):
    net_config = ''
    if caffe_flag:
        net_config = caffe.proto.caffe_pb2.NetParameter()
    else:
        net_config = caffe_parse.caffe_pb2.NetParameter()
    return readProtoFile(filepath, net_config)

def readProtoSolverFile(filepath):
    solver_config = ''
    if caffe_flag:
        solver_config = caffe.proto.caffe_pb2.SolverParameter()
    else:
        solver_config = caffe_parse.caffe_pb2.SolverParameter()
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
    param_string = "num_filter=%d, pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d), use_bias=%s" %\
        (param.num_output, pad, pad, kernel_size,\
        kernel_size, stride, stride, param.bias_term)
    # deal with dilation. Won't be in deconvolution
    if dilate > 1:
        param_string += ", dilate=(%d, %d)" % (dilate, dilate)
    return param_string

# For Conv and Dense layers, convert weight_filler and bias_filler into
# w_specs and b_specs of singa
def wbParamToString(layer):
    print layer
    w_specs = ''
    b_specs = ''
    if layer.type == 'Convolution':
        param = layer.convolution_param
    elif layer.type == 'InnerProduct':
        param = layer.inner_product_param
    if len(layer.param) > 0:
        w_specs = '''w_specs = {'init': '%s', 'mean': %d, 'std': %f, 'lr_mult': %f}''' %\
            (param.weight_filler.type, param.weight_filler.mean, \
            param.weight_filler.std, layer.param[0].lr_mult)
        if len(layer.param) > 1:
            b_specs = '''b_specs = {'init': '%s', 'mean': %d, 'std': %f, 'lr_mult': %f}''' %\
                (param.weight_filler.type, param.weight_filler.mean, \
                param.weight_filler.std, layer.param[1].lr_mult)
        else:
            b_specs = '''b_specs = {'init': '%s', 'mean': %d, 'std': %f}''' %\
                (param.weight_filler.type, param.weight_filler.mean, \
                param.weight_filler.std)
    else:
        w_specs = '''w_specs = {'init': '%s', 'mean': %d, 'std': %f}''' %\
            (param.weight_filler.type, param.weight_filler.mean, \
            param.weight_filler.std)

    return w_specs, b_specs

# Convert caffe model into a singa model
def net2script(net_file):
    net = readProtoNetFile(net_file)
    flatten_count = 0
    layer = ''
    # class of loss: 
    #   0 - no loss function
    #   1 - SoftmaxCrossEntropy
    #   2 - SquaredError
    loss = 0
    # class of metric:
    #   0 - no metric
    #   1 - Accuracy 
    metric = 0 
    if len(net.layer):
        layer = net.layer
    elif len(net.layers):
        layer = net.layers
    else:
        raise Exception('Invalid proto file.')
    # Get input size to network
    #input_dim = [1, 3, 224, 224] # default
    #input_dim = layer[0].input_param.shape[0].dim
    #print "Shape of input tensor: ", input_dim

    #TODO: generate input tensors based on prototxt file

    #if hasattr(layer[0], top):
    #    input_name = layer[0].top[0]
    #else:
    #    input_name = layer[0].name

    # record layer mapping from caffe to singa
    #mapping = {input_name : 'data'}
    #need_flatten = {input_name : False}
    output = ''

    for i in range(len(layer)):
        type_string = ''
        param_string = ''
        # replace '-' or '/' by '_'
        name = re.sub('[-/]', '_', layer[i].name)
        if layer[i].type == 'Data' or layer[i].type == 5:
            # Ignore data layer(s)
            continue
        if layer[i].type == 'Convolution' or layer[i].type == 4:
            type_string = 'layer.Conv2D'
            param_string = convParamToString(layer[i].convolution_param)
            w_specs, b_specs = wbParamToString(layer[i])
            param_string += ', ' + w_specs + ', ' + b_specs
        if layer[i].type == 'Pooling' or layer[i].type == 17:
            type_string = 'layer.Pooling2D'
            param = layer[i].pooling_param
            param_string += "pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d)" %\
                (param.pad, param.pad, param.kernel_size,\
                param.kernel_size, param.stride, param.stride)
            if param.pool == 0:
                param_string = param_string + ", mode=model_pb2.PoolingConf.MAX"
            elif param.pool == 1:
                param_string = param_string + ", mode=model_pb2.PoolingConf.AVE"
            else:
                raise Exception("Unknown Pooling Method!")
        if layer[i].type == 'ReLU' or layer[i].type == 18:
            type_string = 'layer.Activation'
            param_string = "mode='relu'"
        if layer[i].type == 'LRN' or layer[i].type == 15:
            type_string = 'layer.LRN'
            param = layer[i].lrn_param
            param_string = "size=%d, alpha=%f, beta=%f, k=%f" %\
                (param.local_size, param.alpha, param.beta, param.k)
        if layer[i].type == 'InnerProduct' or layer[i].type == 14:
            type_string = 'layer.Dense'
            param = layer[i].inner_product_param
            param_string = "num_output=%d, use_bias=%s" % (param.num_output, param.bias_term)
            w_specs, b_specs = wbParamToString(layer[i])
            param_string += ', ' + w_specs + ', ' + b_specs
        if layer[i].type == 'Dropout' or layer[i].type == 6:
            type_string = 'layer.Dropout'
            param = layer[i].dropout_param
            param_string = "p=%f" % param.dropout_ratio
        if layer[i].type == 'Softmax' or layer[i].type == 20:
            type_string = 'layer.Softmax'
            param = layer[i].softmax_param
            param_string = "axis=%d" % param.axis
        if layer[i].type == 'Flatten' or layer[i].type == 8:
            type_string = 'layer.Flatten'
            param_string = ''
        if layer[i].type == 'Accuracy' or layer[i].type == 1:
            type_string = 'metric.Accuracy'
            metric = 1
        if layer[i].type == "SoftmaxWithLoss" or layer[i].type == 21:
            type_string = 'loss.SoftmaxCrossEntropy'
            loss = 1
        if layer[i].type == 'EuclideanLoss' or layer[i].type == 7:
            type_string = 'loss.SquaredError'
            loss = 2

        # Caffe don't support BN officially
        #if layer[i].type == 'BatchNorm':
        #    type_string = 'singa.layer.BatchNormalization'
        #    param = layer[i].batch_norm_param

        if type_string == '':
            raise Exception('Unknown Layer %s!' % layer[i].type)

        #TODO: conversion of Split, Concat and Crop layers of Caffe

        if layer[i].type == 'InnerProduct':
            flatten_name = "flatten_%d" % flatten_count
            output += "\tnet.add(layer.Flatten(name='%s'))\n" % flatten_name
            flatten_count += 1
        if layer[i].type != 'Accuracy' and layer[i].type != 'SoftmaxWithLoss' and layer[i].type != 'EuclideanLoss':
            output += "\tnet.add(%s(%s))\n" % (type_string, '\'' + name + '\', ' + param_string)

        if metric == 1:
            if loss == 1:
                output = '\tnet = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())\n' + output
            elif loss == 2:
                output = '\tnet = ffnet.FeedForwardNet(loss.SquaredError(), metric.Accuracy())\n' + output

    return output

def solver2script(solver_file):
    solver = readProtoSolverFile(solver_file)
    output = ''
    if solver.solver_mode == 1:
        output = "\tlayer.engine = 'singacuda'\n"
    else:
        output = "\tlayer.engine = 'singacuda'\n"

    return output

def caffe2singa(net_file, solver_file, out_name):
    script = open(out_name + '.py', 'w')
    net = net2script(net_file)
    solver = solver2script(solver_file)
    script.write('''# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import layer
from singa import metric
from singa import loss
from singa import net as ffnet

def create_net():
''')
    script.write(solver)
    script.write(net)
    script.close()

def main():
    parser = argparse.ArgumentParser(description='Caffe prototxt to SINGA model parameter converter.\
                    Note that only basic functions are implemented. You are welcomed to contribute to this file.')
    parser.add_argument('net_prototxt', help='The prototxt file for net in Caffe format')
    parser.add_argument('solver_prototxt', help='The prototxt file for solver in Caffe format')
    parser.add_argument('save_model_name', help='The name of the output model prefix')
    args = parser.parse_args()

    if len(sys.argv) == 4:
        caffe2singa(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print 'Few arguments!'

if __name__ == '__main__':
    main()
