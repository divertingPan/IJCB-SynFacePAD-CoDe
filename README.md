# IJCB-SynFacePAD-CoDe
Our Solution for IJCB SynFacePAD 2023 Competition (rank 3rd), reference: SynFacePAD 2023: Competition on Face Presentation Attack Detection Based on Privacy-aware Synthetic Training Data

Our CoDe model is quite simple, due to a heavy overfitting we observed when utilizing bigger model. CoDe-Lc and CoDe-Lh (as
shown in Figure) are two ensemble models consisting of
dual branches using AlexNet as backbone architecture.
Both models were trained from scratch, utilizing a weighted
sampling which was performed to ensure a bona fide-attack
ratio of 1:1. For CoDe-Lc, the cosine similarity function
was employed as the loss function to measure the discrepancy
between the feature layers from each branch. Additionally,
the Mean Squared Error (MSE) loss was used as an
auxiliary metric to evaluate the similarity of features. The
BCE loss was computed for the final prediction as well as each branch’s prediction. The total loss was calculated as
the cumulative sum of all the aforementioned losses. For
CoDe-Lh, the cosine similarity was replaced with the hypersphere
loss (Z. Li, H. Li, K.-Y. Lam, and A. C. Kot. Unseen face presentation
attack detection with hypersphere loss. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), pages 2852–2856, 2020.). The input images were resized to dimensions
of 224×224, and data augmentation techniques were
applied, including random horizontal flipping, scaling and
rotating, gamma adjustment, RGB shifting, and color jittering.
Moreover, for CoDe-Lh, additional augmentation was
introduced by applying random Gaussian blur. The Adam
optimizer with a learning rate of 1e-4 and weight decay of
5e-4 was utilized, along with an exponential learning scheduler
with a gamma value of 0.998. The batch size during
training was set to 128, and the number of training epoch
was defined as 200.

