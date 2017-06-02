class Chamfer_Stats():
    experiments_ids = [4, 5, 6, 7, 8]

    experiment_id_to_bneck = {4: 64, 5: 256, 6: 512, 7: 32, 8: 128}

    best_epochs = {32: (980, 0.00023145349890876625),
               64: (970, 0.00020686166095647683),
               128: (970, 0.00020017094667833303),
               256: (1000, 0.0002053646348071166),
               512: (1000, 0.00019155694878900324)}

    # This is epoch where each architecture reached the training loss achieved by the worse performing architecture (i.e., for the one with the 32-bottleneck layer).
    max_min_epochs = {32: (980, 0.000231453498909),
                  64: (470, 0.000232941098663),
                  128: (380, 0.000232541767805),
                  256: (470, 0.000232944812497),
                  512: (350, 0.000231615319654)}


class EMD_Stats():
    experiments_ids = [10, 11, 12, 13, 14]

    experiment_id_to_bneck = {10: 32, 11: 64, 12: 128, 13: 256, 14: 512}

    best_epochs = {32: (980, 0.00023145349890876625),
                   64: (970, 0.00020686166095647683),
                   128: (970, 0.00020017094667833303),
                   256: (1000, 0.0002053646348071166),
                   512: (1000, 0.00019155694878900324)}

    # This is epoch where each architecture reached the training loss achieved by the worse performing architecture (i.e., for the one with the 32-bottleneck layer).
    max_min_epochs = {32: (980, 0.000231453498909),
                      64: (470, 0.000232941098663),
                      128: (380, 0.000232541767805),
                      256: (470, 0.000232944812497),
                      512: (350, 0.000231615319654)
                      }
