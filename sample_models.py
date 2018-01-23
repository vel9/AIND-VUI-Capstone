from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Dropout,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    gru_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_gru_rnn')(gru_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layers, each with batch normalization
    rnn_1 = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn_1')(input_data)
    # Add batch normalization
    bn_rnn_1 = BatchNormalization(name='bn_rnn_1')(rnn_1)
    
    recur_rnn_input = bn_rnn_1
    recur_rnn_output = bn_rnn_1
    for i in range(recur_layers - 1):
        layer_num = i + 2
        rnn_layer_name = 'rnn_' + str(layer_num)
        rnn_out = GRU(units, activation='relu',
            return_sequences=True, implementation=2, name=rnn_layer_name)(recur_rnn_input)
        bnn_layer_name = 'bn_rnn_' + str(layer_num)
        recur_rnn_output = BatchNormalization(name=bnn_layer_name)(rnn_out)
        recur_rnn_input = recur_rnn_output
        
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(recur_rnn_output)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn'))(input_data)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def op_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    # trying out a model with tanh activations instead of relu
    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn_1 = Bidirectional(GRU(units, return_sequences=True, 
        implementation=2, name='bidir_rnn_1'))(input_data)
    bidir_bn_1 = BatchNormalization(name='bidir_bn_1')(bidir_rnn_1)
    bidir_rnn_2 = Bidirectional(GRU(units, return_sequences=True, 
        implementation=2, name='bidir_rnn_2'))(bidir_bn_1)
    bidir_bn_2 = BatchNormalization(name='bidir_bn_2')(bidir_rnn_2)
    
    time_dense = TimeDistributed(Dense(output_dim))(bidir_bn_2)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def op_model_2(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    
    Same as the CNN model but with Dropout
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_conv_1d = BatchNormalization(name='bn_conv_1d')(conv_1d)
    drop_conv_1d = Dropout(0.5, name='drop_conv_1d')(bn_conv_1d)
    # Add a recurrent layer
    gru_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='gru_rnn', dropout=0.5)(drop_conv_1d)
    bn_rnn = BatchNormalization(name='bn_rnn')(gru_rnn)
    # Add a dense layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def op_model_3(input_dim, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    filters = 200
    kernel_size = 11
    conv_stride = 2
    conv_border_mode = 'valid'
                
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_conv_1d = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add dropout
    drop_conv_1d = Dropout(0.5, name='drop_conv_1d')(bn_conv_1d)
    # Add max pool https://keras.io/layers/pooling/
    max_pooling_1d = MaxPooling1D(name='max_pooling_1d')(drop_conv_1d)
    # Add 1 Bidirectional recurrent layers
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, dropout=0.5), name='bidir_rnn',)(max_pooling_1d)
    bidir_bn = BatchNormalization(name='bidir_bn')(bidir_rnn)
    # Add a dense layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_bn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)/2 # because we're doing max pool
    print(model.summary())
    return model

def final_model(input_dim, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    filters = 200
    kernel_size = 11
    conv_stride = 2
    conv_border_mode = 'valid'
                
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_conv_1d = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add dropout
    drop_conv_1d = Dropout(0.2, name='drop_conv_1d')(bn_conv_1d)
    # Add 2 Bidirectional recurrent layers
    bidir_rnn_1 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, dropout=0.2), name='bidir_rnn_1')(drop_conv_1d)
    bidir_bn_1 = BatchNormalization(name='bidir_bn_1')(bidir_rnn_1)
    bidir_rnn_2 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, dropout=0.2), name='bidir_rnn_2')(bidir_bn_1)
    bidir_bn_2 = BatchNormalization(name='bidir_bn_2')(bidir_rnn_2)
    # Add a dense layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_bn_2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model