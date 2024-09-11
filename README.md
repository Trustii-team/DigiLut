![trustii logo](/HeroImage.png)

This repository contains the [DigiLut](https://www.trustii.io/post/join-the-digilut-challenge-advancing-lung-transplant-rejection-detection-1) data challenge results, including the best notebooks, source code, and documentation.

# The Challenge Background

The DigiLut Challenge has been organized by the Foch Hospital, in partnership with the Health Data Hub and financed by the Banque Publique d’Investissement (Bpifrance). The challenge has been led by Pr. Antoine Roux from Foch Hospital, who is a pneumologist and holds a PhD in immunology. The challenge focuses on the detection of **Graft Rejection Following Lung Transplantation**.

The goal of the DigiLut challenge was to develop algorithms capable of identifying pathological zones in **digitized transbronchial biopsy slides**, determining the presence and severity of graft rejection, based on a unique image bank of digitized graft biopsies from lung transplant patients followed at Foch Hospital.

The aim was to apply state-of-the-art deep learning approaches to the analysis of digitized biopsy slides and pathology characterization, developing an algorithm capable of generating region-of-interest boxes for type A lesions on new, non-annotated slides, using both the annotated and non-annotated data provided.

# The Dataset 

The dataset has been provided by Foch Hospital, constructed and anonymized from digitized biopsy slides. It includes annotations of the zones of interest, created by an international panel of expert pathologists. This database has been made available to competitors who participated in the challenge to create an algorithm to detect graft rejection and its severity.

# The Metric of Evaluation

In this challenge, lesion detection models were evaluated using metrics that balance precision and recall, with particular attention to minimizing false negatives. The chosen metrics are the **Generalized Intersection over Union (GIoU) and the F2 Score**.

# The Development Environment 

Competitors were provided with a managed Jupyterhub environment to access the dataset. Each team had access to a notebook session with 6 vCPU, 13GB RAM, and 521GB persistent storage.

# The Challenge Results

* 256 participants from multiple countries around the world
* 1202 submissions during the 2 months of the challenge
* Best GIoU / F1 score: 0.397 on the private portion of the test set

# Best Notebooks

##### Note: The final score is based on the private leaderboard, which has been calculated using the F1 score, based on a cross-annotated validation test set.

| Ranking    | Team               | Score Public Leaderboard | **Score Private Leaderboard** | Modeling Approach |
|:----------:|:------------------:|:-------------------------:|:----------------------------:|:-----------------:|
| 1          | CVN Team (Loïc Le Bescond and Aymen Sadraoui) | 0.412 | 0.397 | The team from the CVN Lab at INRIA Paris-Saclay approached the DigiLut Data Challenge by focusing on lung histopathology, leveraging their expertise in biomedical image analysis. They began with data preprocessing, using their custom library, prismtoolbox, to extract 512x512 patches from Whole Slide Images (WSIs), distinguishing between positive and negative patches based on the presence_of_lesions column. They implemented a 5-Fold Cross-Validation strategy to ensure balanced class distribution. Their modeling approach involved data augmentation techniques and fine-tuning a pretrained model by Ozan Ciga et al. They employed Weighted Cross Entropy with label smoothing and used the Adam optimizer. After training each fold for 20 epochs on NVIDIA A100-SXM4-80GB GPUs, they ensembled the models using arithmetic mean for final predictions. Lastly, they grouped positive patches into clusters to generate final bounding box coordinates for predictions. |
| 2          | MaD Lab Team (Emmanuelle Salin and Arijana Bohr) | 0.377 | 0.326 | The team led by Arijana Bohr and Emmanuelle Salin developed a diagnostic algorithm to detect graft rejection in lung transplant patients by analyzing transbronchial biopsy images. Their methodology involved data preprocessing, artifact detection, and model training using a lightweight DINO vision transformer. They began by segmenting high-resolution Whole Slide Images (WSIs) into manageable patches of 224x224 pixels, focusing on magnification levels two and three for optimal balance between detail and context. Preprocessing steps included removing irrelevant background regions and artifacts, enhancing contrast, and normalizing image patches. The DINO model, pre-trained on ImageNet and refined with soft labels and focal loss, was trained to classify these patches. To enhance accuracy, they employed a sliding window approach and clustering techniques during inference, refining predictions at higher magnifications. The model's performance was evaluated using the Generalized Intersection over Union (GIoU) metric. Despite computational constraints, their approach showed promise in improving early detection of acute rejection in lung transplant patients. |
| 3          | Harshit Sheoran | 0.346 | 0.250 | The modeling approach taken by Harshit Sheoran for the DigiLut Data Challenge 2024 is a sophisticated two-part strategy involving ranking and detection. Initially, the massive Whole Slide Images (WSIs) are divided into smaller units called "structures." A ranking model is employed to predict whether each structure contains a bounding box (bbox). Structures identified as containing bboxes are then passed to the detection model, which makes precise bbox predictions. During the training phase, pseudo-labels are generated from unlabeled data to enhance the model's training. For inference, the model preprocesses the WSI into structures and uses both the ranking and detection models to average predictions, selecting the top structures. Soft Non-Maximum Suppression (soft-NMS) is applied to the detection predictions for final bbox outputs. The technical implementation involves multiple folds from CoAT and MaxViT models for ranking and Co-DINO models for detection, utilizing high-performance hardware, including an AMD TR 3960X CPU, 128GB RAM, and four NVIDIA 3090 GPUs. |

For more details, check out each winning solution report and source code in the 'repository' above.

You can also check details of each solution by chatting with the following Gen AI models trained on each winner's code and documentation:

| Ranking    | Team               | Understand Tech Model URL |
|:----------:|:------------------:|:-------------------------:|
| 1          | CVN Team (Loïc Le Bescond and Aymen Sadraoui) | [Link](https://app.understand.tech/?api_key=7dbafc1ce66d8ceb48d955880a87bea949cd6f416164fc0882ddbd8408d25445&model_id=first%20solution%20digilut) | 
| 2          | MaD Lab Team (Emmanuelle Salin and Arijana Bohr) | [Link](https://app.understand.tech/?api_key=cb910c5aa3739436f4da4e678472da616641d049bc43d4e6dd15a3561cb295a4&model_id=Second%20Team%20Digilut) | 
| 3          | Harshit Sheoran | [Link](https://app.understand.tech/?api_key=c1a8577dfa307ca732a5179f304b9232776056d511e9062497d7d624d0c999c2&model_id=third%20solution%20digilut) | 

# More Information

If you are interested in accessing the dataset or collaborating with Trustii.io and Foch Hospital on this project, please reach out to us at contact@trustii.io.

To access the challenge forum discussions, the dataset description, and the complete code source of all participants, check out the challenge webpage at [Trustii.io](https://app.trustii.io).