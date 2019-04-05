# REV 1.5

Same architecture as REV 1, but where the loss is the K.mean(recon * x + (kl * (1-x))).
Previously, the K.mean was left off erroneously.
