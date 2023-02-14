import soundata

# 版本太低，没有支持 ，放弃当前选项

# learn wich datasets are available in soundata
print(soundata.list_datasets())

# choose a dataset and download it
dataset = soundata.initialize('urbansound8k', data_home='choose_where_data_live')
dataset.download()

# get annotations and audio for a random clip
example_clip = dataset.choice_clip()
tags = example_clip.tags
y, sr = example_clip.audio