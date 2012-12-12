PSLRHmm
=======

A parallel sparse left-right hidden markov model c++ library.

Features:
- Parallel: structures are thread safe and Baum-Welch training is parallelized via OpenMP.
- Sparse: Emission probabilities and transition probabilities are stored in sparse vectors to enable large possible alphabets and large non-ergodic HMMs
- Left-right: most C++ HMM libraries seem to support only Baum-Welsh training with repeating models. However, many times we want to train on many non-repeating examples. This library supports B-W for left-right as specified by Rabiner
- C++: for speed and readability... templated for speed and non-readability
- Library: for usability beyond me

For now, library supports only discrete emissions.
