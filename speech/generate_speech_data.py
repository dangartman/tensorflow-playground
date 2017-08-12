from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import numpy as np
import subprocess
import random

DATA_DIR = 'data/'

NUMBERS_PATH = DATA_DIR + "spoken_numbers"
WORDS_PATH = DATA_DIR + "spoken_words_wav"
SENTENCES_PATH = DATA_DIR + "spoken_sentences_wav"
SENTENCES_MLL_PATH = DATA_DIR + "spoken_sentences_mll_wav"

good_voices = {
    'english-mb-en1': {'name': 'En1', 'rate': 100},
    'us-mbrola-1': {'name': 'Us1', 'rate': 120},
    'us-mbrola-2': {'name': 'Us2', 'rate': 120},
    'us-mbrola-3': {'name': 'Us3', 'rate': 120},
    'en-german': {'name': 'German', 'rate': 110},
    'en-german-5': {'name': 'German1', 'rate': 100},
    'en-romanian': {'name': 'Romanian', 'rate': 120},
    'en-dutch': {'name': 'Dutch', 'rate': 120},
    'en-french': {'name': 'French', 'rate': 110},
    'en-hungarian': {'name': 'Hungarian', 'rate': 100},
    'en-swedish': {'name': 'Swedish', 'rate': 110},
    'en-swedish-f': {'name': 'Swedish1', 'rate': 110}
}

bad_voices = {
    'english-us': {'name': 'Us', 'rate': 120},
    'en-greek': {'name': 'Greek', 'rate': 150},
    'english': {'name': 'En', 'rate': 120},
    'english-north': {'name': 'En2', 'rate': 130},
    'english_rp': {'name': 'En3', 'rate': 110},
    'english_wmids': {'name': 'En4', 'rate': 120},
    'en-scottish': {'name': 'Scottish', 'rate': 130},
    'en-westindies': {'name': 'Westindies', 'rate': 140},

    'en-afrikaans': {'name': 'Afrikaans', 'rate': 100},
    'en-polish': {'name': 'Polish', 'rate': 110}
}

validation_percent = 10
validation_voices = ['us-mbrola-2', 'en-german-5']
n_features = 26


def check_voices():
    voice_infos = str(subprocess.check_output(["espeak", "--voices=en"])).split("\n")[1:-1]
    voices = map(lambda x: x.split()[3], voice_infos)
    for voice in good_voices.keys():
        if voice in voices:
            print(voice + " FOUND!")
    for voice in good_voices.keys():
        if not voice in voices:
            print(voice + " MISSING!")
            del good_voices[voice]


def generate_mfcc(voice_name, voice_id, line, line_num, rate, path):
    from librosa import load
    from scikits.talkbox.features import mfcc

    filename = path + "/wav/{0}_{1}_{2}.wav".format(line_num, voice_name, rate)
    try:
        out = str(subprocess.check_output([
            "espeak",
            "-v", voice_id,
            "-w", filename,
            "-s {0}".format(rate),
            line
        ], stderr=subprocess.STDOUT))
        if "FATAL ERROR" in out:
            print("CANNOT GENERATE WAV")
        else:
            signal, sample_rate = load(filename, mono=True)
            mel_features, mspec, spec = mfcc(signal, fs=sample_rate, nceps=n_features)
            # mel_features = np.swapaxes(mel_features, 0, 1)  # timesteps x nFeatures -> nFeatures x timesteps
            np.save(path + "/mfcc/%s_%s_%d.npy" % (line_num, voice_name, rate), mel_features)
    except:
        pass


def generate_labels(line, path, line_num, relevant_words):
    num_of_labels = len(relevant_words) + 1  # Add last label if none words are relevant
    labels = np.full(num_of_labels, -1)
    at_least_one_present = False
    for word in line.split(" "):
        try:
            relevant_index = relevant_words.index(word)
            labels[relevant_index] = 1
            at_least_one_present = True
        except:
            pass  # ignore if word is not relevant
    if not at_least_one_present:
        labels[num_of_labels - 1] = 1

    np.save(path + "/labels/%s.npy" % line_num, labels)
    return labels


def generate_phonemes(line, path):
    pronounced = subprocess.check_output(["./line_to_phonemes", line]).decode('UTF-8').strip()  # todo
    # phonemes = string_to_int_line(pronounced, pad_to=max_line_length)  # hack for numbers!
    # phonemes = string_to_int_line(line, pad_to=max_line_length)
    # np.save(path + "/phonemes/%s.npy" % line, phonemes)


def generate(lines, path, relevant_words = None):
    # generate a bunch of files for each line (with many voices, nuances):
    # spoken wav
    # mfcc: Mel-frequency cepstrum
    # mll labels
    if not os.path.exists(path): os.mkdir(path)
    if not os.path.exists(path + "/labels/"): os.mkdir(path + "/labels/")
    if not os.path.exists(path + "/mfcc/"): os.mkdir(path + "/mfcc/")
    if not os.path.exists(path + "/wav/"): os.mkdir(path + "/wav/")
    out = open(path + "/lines.list", "wt")
    line_num = 1
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode('UTF-8').strip()
        type = "train"
        if random.randint(1, 100) < validation_percent:
            type = "validation"
        print("generating [%s] %s" % (type, line))
        out.write("%d:%s:%s\n" % (line_num, type, line))
        voices = good_voices.keys()
        if relevant_words:
            generate_labels(line, path, line_num, relevant_words)
            if type == "validation":
                voice_id = validation_voices[random.randint(0, len(validation_voices) - 1)]
            else:
                voice_id = voices[random.randint(0, len(voices) - 1)]
                while voice_id in validation_voices:
                    voice_id = voices[random.randint(0, len(voices) - 1)]
            voices = [voice_id]
        for voice in voices:
            # from_rate = good_voices[voice]['rate'] - 40
            # to_rate = good_voices[voice]['rate'] + 81
            # for rate in range(from_rate, to_rate, 20):
            rate = random.randint(good_voices[voice]['rate'] - 30, good_voices[voice]['rate'] + 40)
            try:
                generate_mfcc(good_voices[voice]['name'], voice, line, line_num, rate, path)
            except:
                pass  # ignore after debug!
        line_num += 1


def generate_lines(relevant_words, irrelevant_words, num_of_lines, max_line_length, mean_relevance_percent):
    lines = []
    for i in range(0, num_of_lines):
        line = ""
        for w in range(0, random.randint(1, max_line_length)):
            if random.randint(1, 100) < mean_relevance_percent:
                line += relevant_words[random.randint(0, len(relevant_words) - 1)] + " "
            else:
                line += irrelevant_words[random.randint(0, len(irrelevant_words) - 1)] + " "
        lines.append(line)
    return lines


def generate_spoken_numbers():
    nums = list(map(str, range(0, 10)))
    generate(nums, NUMBERS_PATH)


def generate_spoken_words():
    wordslist = "wordslist.txt"
    words = open(wordslist).readlines()
    generate(words, WORDS_PATH)


def generate_spoken_sentences():
    linelist = "sentences.txt"
    lines = open(linelist).readlines()
    generate(lines, SENTENCES_PATH)


def generate_spoken_sentences_mll():
    relevant_wordlist = "mll_relevant_words.txt"
    relevant_words = list(map(
        lambda w: w.replace("\n", ''),
        open(relevant_wordlist).readlines()
    ))
    irrelevant_wordlist = "mll_irrelevant_words.txt"
    irrelevant_words = list(map(
        lambda w: w.replace("\n", ''),
        open(irrelevant_wordlist).readlines()
    ))
    lines = generate_lines(relevant_words, irrelevant_words,
                           num_of_lines=10000, max_line_length=20, mean_relevance_percent=20)
    generate(lines, SENTENCES_MLL_PATH, relevant_words)


def main():
    check_voices()
    generate_spoken_sentences_mll()


if __name__ == '__main__':
    main()
    print("DONE!")
