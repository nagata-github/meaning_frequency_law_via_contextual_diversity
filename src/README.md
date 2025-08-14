### Contents
cal_kappa_with_freq.py: program to calculate kappa and frequency of words from a corpus.

draw_graph.py: program to draw a graph of frequency and context variation.

draw_util.py: utils to draw graphs.

util.py: utils to calculate kappa

### How to use
1. Calculate kappa and frequency of words. Target languages are those in 'bert-base-multilingual-uncased'. They are automatically recoginzed by the BERT model. The tokens in the input corpus should be separated by blank (i.e., pre-tokenized).
    
    `python cal_kappa_with_freq.py INPUT_FILE > OUTPUT_FILE`

     INPUT_FILE: corpus file consisting of sentences (one sentence per line of which tokens are split by blank).
   
     OUTPUT_FILE: result file consisting of token, kappa, and frequeny separated by TAB.
   
    e.g.,
     `python cal_kappa_with_freq.py sample_corpus.txt > output.dat` (for English text) 

      `python cal_kappa_with_freq.py -m tohoku-nlp/bert-base-japanese-v3 sample_corpus_jp.txt > output.dat` (for Japanese text)


3. Draw a graph of frequency and kappa.
   
    `python draw_graph.py -f 10 -b 10 output.data`
   
      -f: frequency threshold
      -b: bin size
