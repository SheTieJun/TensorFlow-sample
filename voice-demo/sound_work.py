import random
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf

from keras.layers import Conv1D, Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

def snr_mixer(clean_file, noise_file, sample_rate=16000):
    clean_audio, _ = librosa.load(clean_file, sr=sample_rate)
    noise_audio, _ = librosa.load(noise_file, sr=sample_rate)
    snr = np.random.randint(-5, 20)
    speech, noise_ori, noisy = generate_mix_audio(clean_audio, noise_audio, snr)
    return speech, noise_ori, noisy
    
# def data_generator():
#     for file in self.clean_files:
#         idx = random.randint(0, len(self.noise_files) - 1)
#         noise_file = self.noise_names[idx]
#         speech, noise_ori, noisy = snr_mixer(file, noise_file, snr)
#         ......
#         for index in range(num_blocks):
#             noisy_speech = noisy[.....]
#             clean_speech = speech[.....]
#             yield noisy_speech.astype('float32'), clean_speech.astype['float32']



def generate_mix_audio(clean_audio, noise_audio, snr):
    # 对两个音频数据进行 padding 以达到相同的长度
    length = max(len(clean_audio), len(noise_audio))
    clean_audio = np.pad(clean_audio, (0, length - len(clean_audio)), 'constant')
    noise_audio = np.pad(noise_audio, (0, length - len(noise_audio)), 'constant')
    
    # 计算出噪音数据的能量
    noise_energy = np.sum(np.square(noise_audio))
    # 计算出噪音数据需要放大的倍数
    target_energy = np.sum(np.square(clean_audio)) / (10**(snr / 10))
    scale = np.sqrt(target_energy / noise_energy)
    
    # 得到混合的音频数据
    mix_audio = clean_audio + noise_audio * scale
    return mix_audio

def data_generator2(clean_files, noise_files, block_len):
    while True:
        for file in clean_files:
            idx = random.randint(0, len(noise_files) - 1)
            noise_file = noise_files[idx]
            speech, noise_ori, noisy = snr_mixer(file, noise_file)
            num_blocks = len(noisy) // block_len
            for index in range(num_blocks):
                start = index * block_len
                end = start + block_len
                noisy_speech = noisy[start:end]
                clean_speech = speech[start:end]
                yield noisy_speech.astype('float32'), clean_speech.astype('float32')


def snr_mixer(clean_file, noise_file, snr):
    # 读取清晰语音和噪声信号
    clean_speech = load_audio_file(clean_file)
    noise_speech = load_audio_file(noise_file)
    
    # 计算清晰语音和噪声信号的功率
    clean_power = np.mean(clean_speech ** 2)
    noise_power = np.mean(noise_speech ** 2)
    
    # 计算噪声信号的放大系数
    alpha = np.sqrt(clean_power / (noise_power * 10.0 ** (snr / 10.0)))
    
    # 生成有噪声的语音
    noisy_speech = clean_speech + alpha * noise_speech
    
    return clean_speech, noise_speech, noisy_speech



def load_audio_file(filename):
    audio, sample_rate = sf.read(filename)
    return audio, sample_rate

def self_define_loss():
    def loss(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)
    return loss


# 一般来说，block_len 越大(1024 或者2048)，对于噪音信号的处理能力就越强，但是计算代价也会相应增加。
block_len = 2048  #代表了分块长度
clean_files=[""]
noise_files=[""]

dataset = tf.data.Dataset.from_generator(
    data_generator2(clean_files, noise_files, block_len),                     
    (tf.float32, tf.float32), 
    output_shapes=(tf.TensorShape([]),tf.TensorShape([])))

dataset_val = tf.data.Dataset.from_generator(
    data_generator2(clean_files, noise_files, block_len), 
     (tf.float32, tf.float32), 
     output_shapes=(tf.TensorShape([]),tf.TensorShape([])),
    args=None
)

# 帧长（win_len）和帧移（win_shift）
win_len = 1024
win_shift= 512
noisy_audio = Input(batch_shape=(None, None), name='input_1')

# such as 16 or 32
num_filters = 16
filter_len = 128 #64/128
num_units = 32
learning_rate = 0.0005
file_path =  './tf/audiowork'

windows = tf.signal.frame(noisy_audio, win_len, win_shift)
stft_res = tf.signal.rfft(windows)

x = Conv1D(filters=num_filters, kernel_size=filter_len, activation='relu')(stft_res)
x = Dense(units=num_units, activation='relu')(x)

backward_res = LSTM(units=num_units)(x)
forward_res = backward_res[:, ::-1, :]
istft_res = tf.signal.irfft(forward_res)

Output = tf.keras.layers.TimeDistributed(Dense(units=1), name='output')(istft_res)

model = Model(inputs=noisy_audio, outputs=Output)
weights_file = './audiowork.h5'

# Define a ModelCheckpoint callback to save the model's weights
checkpoint_callback = ModelCheckpoint(weights_file, save_weights_only=True)
# Define Loss Function and optimizer
sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
model.compile(loss=self_define_loss(), optimizer=sgd)
model.fit(x=dataset,
          epochs=20,
          callbacks=[checkpoint_callback],
          validation_data=dataset_val)
model.load_weights(weights_file)
weights = model.get_weights()


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with tf.io.gfile.GFile(file_path + '.tflite', 'wb') as f:
      f.write(tflite_model)
