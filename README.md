# CycleGAN with 1D CNN model

INTRODUCTION
------------
Tensorflow (Python) implementation of a Cycle Consistant Adverserial Network(CycleGAN) with a Convolutional Neural Network (CNN) model with Gated activations, Residual connections, dilations and PostNets.

Comments/questions are welcome! Please contact: shreyas.seshadri@aalto.fi

Last updated: 30.10.2018


LICENSE
-------

Copyright (C) 2018 Shreyas Seshadri, Aalto University

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The source code must be referenced when used in a published work.

FILES AND FUNCTIONS
-------------------
train_cycleW.py - CycleGAN implementation using WGAN loss with gradient penalty [2]

model_convNet.py - 1D CNN implementation with gated units, residual connections, potNets and dilations

REFERENCES
---------
[1] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” Proc. ICCV 2017, pp. 2223–2232, 2017.

[2] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, “Improved training of Wasserstein GANs,” in Advances in Neural In- formation Processing Systems 30, , 2017, pp. 5767–5777.
