"""
  Demo Offline Streaming
"""

import pickle

from frame_asr import FrameASR

def main():
  """
    Main
  """
  asr = FrameASR()
  data_list = pickle.load(open('data_list.pickle', 'rb'))
  for signal in data_list:
    pred = asr.transcribe(signal)
    if len(pred.strip()):
      print('"{}"'.format(pred))

if __name__ == "__main__":
  main()
