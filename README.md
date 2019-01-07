# 2017_visual_concepts

- prepare
  - DownloadVgg.py: download vgg-16 trained in 1000 classes of imagenet images.
  - GetDataPath.py: get training images from certain class
  -	Summer_project_on_visual_concepts.pptx: introduction of Alan Yuille's report about VC
  - finetune_baseline.py: fine-tune the VGG downloaded in DownloadVGG.py with my training data,that is,100 classes of objects cropped from original imagenet images, each class has about 300 pictures.
  - imagenet_class_index.json: the description of my data
  - network.py: the structure of VGG
- extract VC
  - extract\_vc/: extract hidden layer features,get visual concepts,prunning and show examples of visual concepts
- evaluate
  - vc_acc_score.py: get the recognition probability with certain pair of VCs (can be easily revised to single or triplet, quadruplet case)
  - vc_combination_score.py: get the recognition probability percentage drop when certain pair of VCs are missing.- 
  - vc_score.py: get the recognition probability percentage drop when certain VC is missing.
  - trip_quad.py: get the probability drop when 3/4 certain VCs are missing. The candidate triplets or quadruplets are created from the combination of top pair VCs generated in vc_combination_score.py
  - vc_share.py: get the relationship between the importance of VC and its shared times 
- analysis
  - draw\_picture/: draw some results
- finetune process
  - generate_images: generate some training and testing images 
  - finetune_aperture_multiple_VC.py: finetune with 1 or 2 apertures opened images
  - network_test.py: test

- utils
  - ProjectUtils.py: some useful utils for VGG
  - feature_extractor.py: extract hidden layer features in VGG
  - temp.py: some test code
  -	testvgg.py: similar to feature-extractor.py, useful when you are tring to apply Gaussian template to the hidden layer.
  - utils.py: some useful utils, especially in preprocessing. the only useful one is process_image

- others
  - checking_redundency.py: use higer layer's vision to delete some redundency in visual concepts.
  - complete_success_combination.py: found that single big aperture is more useful in recognition than several small apertures
