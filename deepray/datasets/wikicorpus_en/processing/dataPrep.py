# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import argparse
import os
import pprint
import subprocess

import BookscorpusTextFormatting
import Downloader
import TextSharding
import WikicorpusTextFormatting


def main(args):
  working_dir = os.environ["DATA_PREP_WORKING_DIR"]

  print("Working Directory:", working_dir)
  print("Action:", args.action)
  print("Dataset Name:", args.dataset)

  if args.input_files:
    args.input_files = args.input_files.split(",")

  hdf5_tfrecord_folder_prefix = (
    "_lower_case_"
    + str(args.do_lower_case)
    + "_seq_len_"
    + str(args.max_seq_length)
    + "_random_seed_"
    + str(args.random_seed)
  )

  directory_structure = {
    "download": working_dir + "/download",  # Downloaded and decompressed
    "extracted": working_dir + "/extracted",  # Extracted from whatever the initial format is (e.g., wikiextractor)
    "formatted": working_dir
    + "/formatted_one_article_per_line",  # This is the level where all sources should look the same
    "sharded": working_dir
    + "/sharded_"
    + "training_shards_"
    + str(args.n_training_shards)
    + "_test_shards_"
    + str(args.n_test_shards)
    + "_fraction_"
    + str(args.fraction_test_set),
    "tfrecord": working_dir + "/tfrecord" + hdf5_tfrecord_folder_prefix,
    "hdf5": working_dir + "/hdf5" + hdf5_tfrecord_folder_prefix,
  }

  print("\nDirectory Structure:")
  pp = pprint.PrettyPrinter(indent=2)
  pp.pprint(directory_structure)
  print("")

  if args.action == "download":
    if not os.path.exists(directory_structure["download"]):
      os.makedirs(directory_structure["download"])

    downloader = Downloader.Downloader(args.dataset, directory_structure["download"])
    downloader.download()

  elif args.action == "text_formatting":
    assert (
      args.dataset != "google_pretrained_weights"
      and args.dataset != "nvidia_pretrained_weights"
      and args.dataset != "squad"
      and args.dataset != "mrpc"
    ), "Cannot perform text_formatting on pretrained weights"

    if not os.path.exists(directory_structure["extracted"]):
      os.makedirs(directory_structure["extracted"])

    if not os.path.exists(directory_structure["formatted"]):
      os.makedirs(directory_structure["formatted"])

    if args.dataset == "wikicorpus_en":
      if args.skip_wikiextractor == 0:
        path_to_wikiextractor_in_container = "python wikiextractor/WikiExtractor.py"
        wikiextractor_command = (
          path_to_wikiextractor_in_container
          + " "
          + directory_structure["download"]
          + "/"
          + args.dataset
          + "/wikicorpus_en.xml "
          + "-b 100M --processes "
          + str(args.n_processes)
          + " -o "
          + directory_structure["extracted"]
          + "/"
          + args.dataset
        )
        print("WikiExtractor Command:", wikiextractor_command)
        wikiextractor_process = subprocess.run(wikiextractor_command, shell=True, check=True)
        # wikiextractor_process.communicate()

      wiki_path = directory_structure["extracted"] + "/wikicorpus_en"
      output_filename = directory_structure["formatted"] + "/wikicorpus_en_one_article_per_line.txt"
      wiki_formatter = WikicorpusTextFormatting.WikicorpusTextFormatting(wiki_path, output_filename, recursive=True)
      wiki_formatter.merge()

    elif args.dataset == "wikicorpus_zh":
      raise NotImplementedError(
        "wikicorpus_zh not fully supported at this time. The simplified/tradition Chinese data needs to be "
        "translated and properly segmented still, and should work once this step is added."
      )
      # if args.skip_wikiextractor == 0:
      #     path_to_wikiextractor_in_container = 'wikiextractor/WikiExtractor.py'
      #     wikiextractor_command = path_to_wikiextractor_in_container + ' ' + directory_structure['download'] + '/' + args.dataset + '/wikicorpus_zh.xml ' + '-b 100M --processes ' + str(args.n_processes) + ' -o ' + directory_structure['extracted'] + '/' + args.dataset
      #     print('WikiExtractor Command:', wikiextractor_command)
      #     wikiextractor_process = subprocess.run(wikiextractor_command, shell=True, check=True)
      #     #wikiextractor_process.communicate()
      #
      # wiki_path = directory_structure['extracted'] + '/wikicorpus_zh'
      # output_filename = directory_structure['formatted'] + '/wikicorpus_zh_one_article_per_line.txt'
      # wiki_formatter = WikicorpusTextFormatting.WikicorpusTextFormatting(wiki_path, output_filename, recursive=True)
      # wiki_formatter.merge()
      #
      # assert os.stat(output_filename).st_size > 0, 'File glob did not pick up extracted wiki files from WikiExtractor.'

  elif args.action == "sharding":
    # Note: books+wiki requires user to provide list of input_files (comma-separated with no spaces)
    if args.dataset == "bookscorpus" or "wikicorpus" in args.dataset or "books_wiki" in args.dataset:
      if args.input_files is None:
        if args.dataset == "bookscorpus":
          args.input_files = [directory_structure["formatted"] + "/bookscorpus_one_book_per_line.txt"]
        elif args.dataset == "wikicorpus_en":
          args.input_files = [directory_structure["formatted"] + "/wikicorpus_en_one_article_per_line.txt"]
        elif args.dataset == "wikicorpus_zh":
          args.input_files = [directory_structure["formatted"] + "/wikicorpus_zh_one_article_per_line.txt"]
        elif args.dataset == "books_wiki_en_corpus":
          args.input_files = [
            directory_structure["formatted"] + "/bookscorpus_one_book_per_line.txt",
            directory_structure["formatted"] + "/wikicorpus_en_one_article_per_line.txt",
          ]

      output_file_prefix = directory_structure["sharded"] + "/" + args.dataset + "/" + args.dataset

      if not os.path.exists(directory_structure["sharded"]):
        os.makedirs(directory_structure["sharded"])

      if not os.path.exists(directory_structure["sharded"] + "/" + args.dataset):
        os.makedirs(directory_structure["sharded"] + "/" + args.dataset)

      # Segmentation is here because all datasets look the same in one article/book/whatever per line format, and
      # it seemed unnecessarily complicated to add an additional preprocessing step to call just for this.
      # Different languages (e.g., Chinese simplified/traditional) may require translation and
      # other packages to be called from here -- just add a conditional branch for those extra steps
      segmenter = TextSharding.NLTKSegmenter()
      sharding = TextSharding.Sharding(
        args.input_files, output_file_prefix, args.n_training_shards, args.n_test_shards, args.fraction_test_set
      )

      sharding.load_articles()
      sharding.segment_articles_into_sentences(segmenter)
      sharding.distribute_articles_over_shards()
      sharding.write_shards_to_disk()

      for _dir in ["train", "test"]:
        if not os.path.exists(directory_structure["sharded"] + "/" + args.dataset + "/" + _dir):
          os.makedirs(directory_structure["sharded"] + "/" + args.dataset + "/" + _dir)
        absolute_dir = directory_structure["sharded"] + "/" + args.dataset
        command = "mv " + absolute_dir + "/*" + _dir + "*.txt" + " " + absolute_dir + "/" + _dir
        mv_process = subprocess.Popen(command, shell=True)

        mv_process.wait()
    else:
      assert False, "Unsupported dataset for sharding"

  elif args.action == "create_tfrecord_files":
    if not os.path.exists(directory_structure["tfrecord"] + "/" + args.dataset):
      os.makedirs(directory_structure["tfrecord"] + "/" + args.dataset)
    if args.vocab_file is None:
      args.vocab_file = os.path.join(working_dir, "vocab.txt")

    for _dir in ["train", "test"]:
      electra_preprocessing_command = "python build_pretraining_dataset.py"
      electra_preprocessing_command += (
        " --corpus-dir=" + directory_structure["sharded"] + "/" + args.dataset + "/" + _dir
      )
      electra_preprocessing_command += (
        " --output-dir=" + directory_structure["tfrecord"] + "/" + args.dataset + "/" + _dir
      )
      electra_preprocessing_command += " --vocab-file=" + args.vocab_file
      electra_preprocessing_command += " --do-lower-case" if args.do_lower_case else " --no-lower-case"
      electra_preprocessing_command += " --max-seq-length=" + str(args.max_seq_length)
      electra_preprocessing_command += " --num-processes=8"
      electra_preprocessing_command += (
        " --num-out-files=" + str(args.n_training_shards)
        if _dir == "train"
        else " --num-out-files=" + str(args.n_test_shards)
      )
      electra_preprocessing_process = subprocess.Popen(electra_preprocessing_command, shell=True)

      electra_preprocessing_process.wait()

  elif args.action == "create_hdf5_files":
    raise NotImplementedError


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Preprocessing Application for Everything BERT-related")

  parser.add_argument(
    "--action",
    type=str,
    help="Specify the action you want the app to take. e.g., generate vocab, segment, create tfrecords",
    choices={
      "download",  # Download and verify mdf5/sha sums
      "text_formatting",  # Convert into a file that contains one article/book per line
      "sharding",  # Convert previous formatted text into shards containing one sentence per line
      "create_tfrecord_files",  # Turn each shard into a TFrecord with masking and next sentence prediction info
      "create_hdf5_files",  # Turn each shard into a HDF5 file with masking and next sentence prediction info
    },
  )

  parser.add_argument(
    "--dataset",
    type=str,
    help="Specify the dataset to perform --action on",
    choices={
      "bookscorpus",
      "wikicorpus_en",
      "wikicorpus_zh",
      "books_wiki_en_corpus",
      "google_pretrained_weights",
      "nvidia_pretrained_weights",
      "mrpc",
      "squad",
      "all",
    },
  )

  parser.add_argument("--input_files", type=str, help="Specify the input files in a comma-separated list (no spaces)")

  parser.add_argument(
    "--n_training_shards", type=int, help="Specify the number of training shards to generate", default=2048
  )

  parser.add_argument("--n_test_shards", type=int, help="Specify the number of test shards to generate", default=2048)

  parser.add_argument(
    "--fraction_test_set",
    type=float,
    help="Specify the fraction (0..1) of the data to withhold for the test data split (based on number of sequences)",
    default=0.1,
  )

  parser.add_argument(
    "--segmentation_method",
    type=str,
    help="Specify your choice of sentence segmentation",
    choices={"nltk"},
    default="nltk",
  )

  parser.add_argument(
    "--n_processes", type=int, help="Specify the max number of processes to allow at one time", default=4
  )

  parser.add_argument(
    "--random_seed", type=int, help="Specify the base seed to use for any random number generation", default=12345
  )

  parser.add_argument("--dupe_factor", type=int, help="Specify the duplication factor", default=5)

  parser.add_argument("--masked_lm_prob", type=float, help="Specify the probability for masked lm", default=0.15)

  parser.add_argument("--max_seq_length", type=int, help="Specify the maximum sequence length", default=512)

  parser.add_argument(
    "--do_lower_case",
    type=int,
    help="Specify whether it is cased (0) or uncased (1) (any number greater than 0 will be treated as uncased)",
    default=0,
  )

  parser.add_argument("--vocab_file", type=str, help="Specify absolute path to vocab file to use)")

  parser.add_argument(
    "--skip_wikiextractor", type=int, help="Specify whether to skip wikiextractor step 0=False, 1=True", default=0
  )

  parser.add_argument(
    "--interactive_json_config_generator",
    type=str,
    help="Specify the action you want the app to take. e.g., generate vocab, segment, create tfrecords",
  )

  args = parser.parse_args()
  main(args)
