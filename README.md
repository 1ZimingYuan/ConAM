# ConAM
The so-called “attention” is an efficient mechanism to improve the performance of convolutional neural networks. It uses contextual information to recalibrate the input to strengthen the propagation of informative features. Most of the existing attention mechanisms use either local or global distribution as contextual information to recalibrate the features directly. We propose a novel attention mechanism that recalibrates the features with the correlation, which we term confidence, between local and global contextual information instead of the only local or global one. The proposed attention mechanism extracts the local and global contextual information simultaneously, and calculates the confidence between them, then uses this confidence to recalibrate the input pixels. The extraction of local and global contextual information enriches the diversity of features. The recalibration with confidence suppresses useless information while enhancing the informative one with fewer parameters. 