"""
  WER
"""

from jiwer import wer

def main():
  """
    Main
  """
  label_file_name = 'label.txt'
  label = open(label_file_name, 'r').readline()
  transcript = open('model_output.pickle')\
      .readlines()[1].split(',')[1]
  print('\n>>>>Label from', label_file_name, 'is :', label)
  print('\n>>>>Transcript:', transcript)
  print('Word Error Rate(WER):', wer(label, transcript))

if __name__ == "__main__":
  main()
