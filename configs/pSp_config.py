# Config params for pSp model
import math

# Output size of generator
# FFHQ = 1024, rest 256
output_size = 1024

# Which encoder to use
encoder_type = 'GradualStyleEncoder'

# Whether to add average latent vector to generate codes from encoder
start_from_latent_avg = True

# Whether to learn in w space instead of w+
# False for FFHQ, True otherwise
learn_in_w = False

# Number of input image channels to the psp encoder
input_nc = 3
label_nc = 0

# compute number of style inputs based on the output resolution
n_styles = int(math.log(output_size, 2)) * 2 - 2
