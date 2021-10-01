# Phys_Seg

Physics-informed segmentation algorithm packaged in an easy-to-use fashion. This software is based on the following publications:

Borges, P., Sudre, C., Varsavsky, T., Thomas, D., Drobnjak, I., Ourselin, S., Car-doso, M.J.: Physics-informed brain MRI segmentation. Lecture Notes in ComputerScience11827 LNCS, 100–109 (2020). https://doi.org/10.1007/978-3-030-32778-111

Borges, P., Shaw, R., Varsavsky, T., Klaser, K., Thomas, D., Drobnjak, I., Ourselin,S.,  Jorge  Cardoso,  M.:  The  Role  of  MRI  Physics  in  Brain  Segmentation  CNNs:Achieving Acquisition Invariance and Instructive Uncertainties. In: Svoboda, D.,Burgos, N., Wolterink, J.M., Zhao, C. (eds.) Simulation and Synthesis in MedicalImaging. pp. 67–76. Springer International Publishing, Cham (2021)

Organisation of this repository is based on that of HD-BET: https://github.com/MIC-DKFZ/HD-BET
# Installation
1. Clone the repository:
```
git clone https://github.com/pedrob37/Phys_Seg/
```
2. Change into the cloned repository directory, and install it:
```
cd Phys_Seg
pip install -e
```

# Usage
Phys-Seg can be run on either a CPU or a GPU (recommended). Phys-Seg requires an input nifti file, and a sequence (MPRAGE only for now), and optionally, relevant sequence parameters:
```
python3 phys_seg.py --input <FILENAME> --sequence <SEQUENCE_CHOICE> --physics_params <PARAMETERS>
```
If sequence parameters are passed to Phys-Seg then a model trained with explicit physics-parameter passing is downloaded and used. Otherwise, a parameter-absent model is downloaded and employed instead.
The physics-absent models while shown not to be as effective as their counterparts, should still provide good generalisability and consistency.
