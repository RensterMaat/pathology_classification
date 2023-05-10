# pathology_classification
WSI Classification based on preextracted features

Todo:
- Make slide_ids into slide_paths; this removes a hyperparameter from config and increases flexibility
- Make exception if slide is not found
- Heatmap plotting separately for different classes

To discuss:
- CLAM uses multibranch attention, whereas HIPT only uses singlebranch attention
- HIPT has a rho linear layer between attention and classifier, whereas CLAM does not. 