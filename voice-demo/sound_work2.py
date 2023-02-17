import tensorflow as tf
from keras.layers import Conv1D, Conv1DTranspose, Input



# input_shape 是指模型输入的张量形状，对于声音降噪任务，通常使用时域上的窗口形式输入模型。
def get_denoising_model(input_shape):
    model_input = Input(shape=input_shape, name='input')
    x = Conv1D(filters=16, kernel_size=21, padding='same', activation='relu')(model_input)
    x = Conv1D(filters=32, kernel_size=21, padding='same', activation='relu')(x)
    x = Conv1DTranspose(filters=32, kernel_size=21, padding='same', activation='relu')(x)
    model_output = Conv1DTranspose(filters=1, kernel_size=21, padding='same', activation='linear')(x)
    model = tf.keras.models.Model(model_input, model_output)
    model.compile(loss='mse', optimizer='adam')
    return model


    