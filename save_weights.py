from keras import backend as K
from keras.models import load_model
from keras.utils.conv_utils import convert_kernel


model = load_model('output/Sprouts_smallvggnet.h5')
model.save_weights('output/Sprouts_weights.h5')
