# RegionalMAE
Welcome to Pulmonary Nodule [RegionalMAE](/RegionalMAE/).

## Table of Contents
- [Abstract:](#abstract)
- [Requirements:](#requirements)
- [Getting Started:](#getting-started)
- [Components:](#components)
- [Citation:](#citation)

## Abstract: 
#### BACKGROUND:
Lung cancer remains the leading cause of cancer-related mortality in the United States, despite the adoption of low-dose computed tomography (LDCT) and updated screening guidelines from the United States Preventive Service Task Force (USPSTF) [19]. Limited infrastructure and financial costs continue to hinder widespread LDCT adoption, while the increasing detection of indeterminate pulmonary nodules (4–20mm) challenges accurate diagnosis and clinical decision-making. Therefore, there is a need for improve lung cancer screening computational tools that work within standard screening protocols.
#### METHODS:
We address these limitations by pretraining masked autoencoders (MAE) on the COPDGene dataset, which captures chronic lung inflammatory disease features. Emphysema and airway disease, two distinct subtypes of COPD, are pathophysiological manifestations of chronic lung inflammation [4, 15]. Incorporating these features may enhance the model’s ability
to distinguish between malignant and benign pulmonary nodules. By exploring multiple masking strategies, we optimize network attention on parenchymal and perinodular features, improving the extraction of relevant image biomarkers. 
#### RESULTS:
Our results demonstrate that pretraining on the COPDGene dataset using random masking (r-masking) achieves superior classification performance, with a sensitivity of 88.79%, specificity of 86.27%, and an AUC of 0.931, when compared to self-pretraining on National Lung Cancer Screening Trial (NLST), and supervised learning on NLST. 
##### CONCLUSION:
This highlights the importance of leveraging chronic disease datasets for self-supervised learning and underscores the potential of MAE-based approaches to improve nodule classification in clinical settings. 

## Requirements:

## Getting Started:

## Components:
### [bin:](/RadiomicConcept/bin/)
  Contains bash, and post-analysis tools for evaluating model performance across modes.
### [Networks:](/RadiomicConcept/networks/)
  Contains the network architecture files, and components.
### [Utilities:](/RadiomicConcept/utils/)
----
## References:
[4] Carr, L.L., Jacobson, S., Lynch, D.A., Foreman, M.G., Flenaugh, E.L., Hersh, C.P., Sciurba, F.C., Wilson, D.O., Sieren, J.C., Mulhall, P., Kim, V., Kinsey, C.M., Bowler, R.P.: Features of COPD as Predictors of Lung Cancer. Chest 153(6), 1326–1335 (Jun 2018). https://doi.org/10.1016/j.chest.2018.01.049
[15] Pai, S., Bontempi, D., Hadzic, I., Prudente, V., Sokač, M., Chaunzwa, T.L., Bernatz, S., Hosny, A., Mak, R.H., Birkbak, N.J., Aerts, H.J.W.L.: Foundation model for cancer imaging biomarkers. Nature Machine Intelligence 6(3), 354–367 (Mar 2024). https://doi.org/10.1038/s42256-024-00807-9, publisher: Nature Publishing Group
[19] US Preventive Services Task Force; Krist, A.H., Davidson, K.W., Mangione, C.M., Barry, M.J., Cabana, M., Caughey, A.B., Davis, E.M., Donahue, K.E., Doubeni, C.A., Kubik, M., Landefeld, C.S., Li, L., Ogedegbe, G., Owens, D.K., Pbert, L., Silverstein, M., Stevermer, J., Tseng, C.W., Wong, J.B.: Screening for Lung Cancer: US Preventive Services Task Force Recommendation Statement. JAMA 325(10), 962–970 (Mar 2021). https://doi.org/10.1001/jama.2021.1117
