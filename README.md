# NID: neural image denoiser
A package for denoising the noisy neural images using trained machine learning models. The supported models are summarized below: 

| name | detail | Python| Matlab|
|---   |---     |---    |---    |
| nn   |  a trained neural network for denoising images [1]| √  |√ |  

## Python
in the folder **/python/**, there is a ipynb file **train_model_with_pinky40** file. This file includes code for model training and testing. 

The trained model were saved into the folder **/python/logs/*.onnx** and **/python/logs/*.pth** for future use. 
## MATLAB 
**required packages**
* [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)
* [Deep Learning Toolbox Converter](https://www.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format)


We created a MATLAB wrapper that loads the trained models (.onnx files) **/matlab/NID.m**. The file **/matlab/demo_nid.m** illustrates how to run the package. 
## Reference
[1]. EASE :EM-AssistedSourceExtraction1fromcalciumimagingdata
## License

Copyright 2018 Pengcheng Zhou

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.



