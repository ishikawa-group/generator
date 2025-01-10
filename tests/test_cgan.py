import sys
sys.path.append("../")

from conditional_gan.cgan import cgan

target_number = 3
# Perform CGAN to get MNIST-line figure of target number
cgan(target_number=target_number)
