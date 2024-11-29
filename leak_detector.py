# Copyright 2024 Adit Anand. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run audio classification."""

import argparse
import time

from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from utils import Plotter


def run(model: str, max_results: int, score_threshold: float,
        overlapping_factor: float) -> None:
  """Continuously run inference on audio data acquired from the device.

  Args:
    model: Name of the TFLite audio classification model.
    max_results: Maximum number of classification results to display.
    score_threshold: The score threshold of classification results.
    overlapping_factor: Target overlapping between adjacent inferences.
  """

  if (overlapping_factor <= 0) or (overlapping_factor >= 1.0):
    raise ValueError('Overlapping factor must be between 0 and 1.')

  if (score_threshold < 0) or (score_threshold > 1.0):
    raise ValueError('Score threshold must be between (inclusive) 0 and 1.')

  classification_result_list = []
  # Initialize a plotter instance to display the classification results.
  plotter = Plotter()

  def save_result(result: audio.AudioClassifierResult, timestamp_ms: int):
    result.timestamp_ms = timestamp_ms
    classification_result_list.append(result)

  # Initialize the audio classification model.
  base_options = python.BaseOptions(model_asset_path=model)
  options = audio.AudioClassifierOptions(
      base_options=base_options, running_mode=audio.RunningMode.AUDIO_STREAM,
      max_results=max_results, score_threshold=score_threshold,
      result_callback=save_result)
  classifier = audio.AudioClassifier.create_from_options(options)

  # Initialize the audio recorder and a tensor to store the audio input.
  # The sample rate may need to be changed to match your input device.
  # For example, an AT2020 requires sample_rate 44100.
  buffer_size, sample_rate, num_channels = 15600, 16000, 1
  audio_format = containers.AudioDataFormat(num_channels, sample_rate)
  record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)
  audio_data = containers.AudioData(buffer_size, audio_format)

  # We'll try to run inference every interval_between_inference seconds.
  # This is usually half of the model's input length to create an overlapping
  # between incoming audio segments to improve classification accuracy.
  input_length_in_second = float(len(
      audio_data.buffer)) / audio_data.audio_format.sample_rate
  interval_between_inference = input_length_in_second * (1 - overlapping_factor)
  pause_time = interval_between_inference * 0.1
  last_inference_time = time.time()

  # Start audio recording in the background.
  record.start_recording()

  faucet_runtime=0
  last_analysis_time=0
  
  # Loop until the user close the classification results plot.
  while True:
    # Wait until at least interval_between_inference seconds has passed since
    # the last inference.
    now = time.time()
    diff = now - last_inference_time
    if diff < interval_between_inference:
      time.sleep(pause_time)
      continue
    last_inference_time = now

    # Load the input audio from the AudioRecord instance and run classify.
    data = record.read(buffer_size)
    # audio_data.load_from_array(data.astype(np.float32))
    audio_data.load_from_array(data)
    classifier.classify_async(audio_data, round(last_inference_time * 1000))

    # # Plot the classification results.
    if classification_result_list:
      #print(classification_result_list)
      plotter.plot(classification_result_list[-1])
      

    if last_analysis_time==0:
      last_analysis_time=now
    diff = now - last_analysis_time
    if diff >= 5:
      #print(classification_result_list,"code")
      faucet_score_sum=0
      score_count=0
      last_analysis_time = now
      for i in range (len(classification_result_list)):
        result=classification_result_list[i]
        if i==0:
          start_time=result.timestamp_ms
        ts=result.timestamp_ms-start_time
        score_count+=2
        for classification in result.classifications:
          for i in range(len(classification.categories)):
            category=classification.categories[i]
            if category.category_name == 'Water tap, faucet' or category.category_name == 'Sink (filling or washing)':
              faucet_score_sum+=category.score
      faucet_score_average=faucet_score_sum/score_count
      #print(faucet_score_sum,score_count,faucet_score_average)
      if faucet_score_average>=0.5:
        faucet_runtime+=5
        print("The faucet has been running for",faucet_runtime,"seconds.")
      else:
        faucet_runtime=0
      classification_result_list.clear()
      if faucet_runtime==600:
        print("ALERT: YOUR FAUCET HAS BEEN RUNNING FOR 10 MINUTES")
    
    

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of the audio classification model.',
      required=False,
      default='yamnet.tflite')
  parser.add_argument(
      '--maxResults',
      help='Maximum number of results to show.',
      required=False,
      default=5)
  parser.add_argument(
      '--overlappingFactor',
      help='Target overlapping between adjacent inferences. Value must be in (0, 1)',
      required=False,
      default=0.5)
  parser.add_argument(
      '--scoreThreshold',
      help='The score threshold of classification results.',
      required=False,
      default=0.0)
  args = parser.parse_args()

  run(args.model, int(args.maxResults), float(args.scoreThreshold),
      float(args.overlappingFactor))


if __name__ == '__main__':
  main()
