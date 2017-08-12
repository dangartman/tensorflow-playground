import tflearn
from speech_data import wave_batch_generator, Target, load_wav_file, path
import numpy

# Simple spoken digit recognition demo, with 98% accuracy in under a minute

# Training Step: 544  | total loss: 0.15866
# | Adam | epoch: 034 | loss: 0.15866 - acc: 0.9818 -- iter: 0000/1000

if __name__ == '__main__':
    batch = wave_batch_generator(10000, target=Target.digits)
    X, Y = next(batch)

    number_classes = 10  # Digits

    # Classification
    tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

    net = tflearn.input_data(shape=[None, 8192])
    net = tflearn.fully_connected(net, 64)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, number_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    model.fit(X, Y, n_epoch=3, show_metric=True, snapshot_step=100)
    # Overfitting okay for now

    demo_file = "5_Vicki_260.wav"
    demo = load_wav_file(path + demo_file)
    result = model.predict([demo])
    result = numpy.argmax(result)
    print("predicted digit for %s : result = %d " % (demo_file, result))
