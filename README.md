PSLRHmm
=======

A parallel sparse left-right hidden markov model c++ library.

Features:
- Parallel: structures will be thread safe and some algorithms will be implemented for parallel computation.
- Sparse: Emission probabilities and transition probabilities will be stored in sparse vectors to enable large alphabets
- Left-right: most C++ HMM libraries seem to support only Baum-Welsh training with repeating models. However, many times we want to train on many non-repeating examples. This library will support B-W for left-right as specified by Rabiner
- C++: for speed and readibility
- Library: for usability

For now, library supports only discrete emissions.
