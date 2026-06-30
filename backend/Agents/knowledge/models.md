# Models

SVM with RBF kernel is a classical nonlinear classifier. It is often strong on small or medium tabular datasets. Important parameters include C and gamma.

Random Forest is a tree ensemble using bagged decision trees. It is robust and interpretable through feature importance.

XGBoost, LightGBM, and CatBoost are gradient boosting models. They often perform very well on tabular data and are strong benchmarks for gait feature classification.

MLP is a feed-forward neural network trained on selected gait features. It learns nonlinear patterns but needs careful tuning.

FT-Transformer is a tabular deep learning model that tokenizes each feature and learns feature interactions through attention.

AutoencoderClassifier compresses gait features into a latent representation, reconstructs the original features, and predicts the class from the latent vector. Its loss combines classification loss and reconstruction loss.