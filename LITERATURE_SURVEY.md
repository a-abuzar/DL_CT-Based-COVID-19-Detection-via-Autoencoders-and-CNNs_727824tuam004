# Literature Survey

This survey reviews 5 notable research papers on deep learning applications for COVID-19 detection using CT scans. It identifies the methodologies used, their results, and the existing gaps that this project aims to address.

## Survey Table

| Paper Title | Year | Method | Result (Accuracy/AUC) | Gap / Limitation |
| :--- | :---: | :--- | :--- | :--- |
| Deep learning-based detection for COVID-19 from chest CT using weak label | 2020 | Pre-trained ResNet50 for lesion localization and classification | Acc: 95.9%, AUC: 0.97 | High computational cost; prone to overfitting on small datasets without structural priors. |
| COVID-Net CT-2: Enhanced Deep Neural Networks for Detection of COVID-19 | 2021 | Custom CNN (COVID-Net) optimized via machine-driven design | Acc: ~96% | Pure supervised learning; struggles with domain shifts if the scanner type changes. |
| Automated detection of COVID-19 cases using deep neural networks with X-ray images | 2020 | DarkNet architecture modified for binary classification | Acc: 98.08% | Primarily focused on X-Rays; CT scans provide 3D context often ignored by simple 2D transfer learning. |
| Anomaly detection-based deep learning approaches for COVID-19 | 2021 | Generative Adversarial Networks (GANs) for anomaly detection | AUC: 0.94 | GANs are notoriously hard to train and suffer from mode collapse; reconstruction is often noisy. |
| Auto-encoder-based feature extraction for COVID-19 CT scan classification | 2022 | Convolutional Autoencoder (CAE) + SVM / Random Forest | Acc: 93.5% | Using traditional ML classifiers (SVM) limits the end-to-end backpropagation capabilities of the model. |

## Gap Analysis
While standard CNNs (like ResNet or COVID-Net) achieve high accuracy, they operate in a purely supervised manner and can overfit to specific scanner noise rather than true pathological features. On the other hand, pure anomaly detection methods (GANs) are unstable to train. Methods utilizing Autoencoders often pair them with traditional ML classifiers (like SVMs), which breaks the end-to-end deep learning pipeline.

**What is missing:** A stable, end-to-end trainable hybrid model that uses an Autoencoder for robust, unsupervised structural feature extraction, seamlessly integrated with a modern, lightweight CNN backend (like ResNet18) for fine-tuned classification. 

## Justification for Chosen Approach
Our approach directly addresses the identified gaps by combining a Convolutional Autoencoder with a ResNet18 classifier in an end-to-end PyTorch pipeline. The Autoencoder forces the network to learn the underlying structural representation of the lungs (reducing overfitting to noise), while the ResNet backend provides powerful hierarchical feature classification. This guarantees stability in training (unlike GANs) and high performance (unlike SVMs).
