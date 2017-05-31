best_epochs = {32: (980, 0.00023145349890876625),
               64: (970, 0.00020686166095647683),
               128: (970, 0.00020017094667833303),
               256: (1000, 0.0002053646348071166),
               512: (1000, 0.00019155694878900324)
               }

# This is epoch where each architecture reached the training loss achieved by the worse performing architecture (i.e., for the one with the 32-bottleneck layer).
max_min_epochs = {32: (980, 0.000231453498909),
                  64: (470, 0.000232941098663),
                  128: (380, 0.000232541767805),
                  256: (470, 0.000232944812497),
                  512: (350, 0.000231615319654)
                  }
