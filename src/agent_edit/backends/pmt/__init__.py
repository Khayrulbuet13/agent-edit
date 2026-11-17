"""PMT (Plain Mask Transformer on frozen DINOv3): the class head over cached queries is editable."""

import torch

# Every CPU op here is tiny (200 x 1024 matmuls): torch's default thread pool makes them much slower
# when several of these processes run at once. One thread also fixes the CPU summation order, so
# scores do not depend on the core count.
torch.set_num_threads(1)
